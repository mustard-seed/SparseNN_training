import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import numpy as np

import custom_modules.custom_modules as modules


def compute_group_lasso_mask(inputTensor: torch.Tensor, clusterSize: int, threshold: float) -> torch.Tensor:
    mask = torch.zeros_like(inputTensor, dtype=torch.float)
    input_dims = inputTensor.size()
    numChannels = input_dims[1]
    N = input_dims[0]

    # Make channel the least significant dimension
    # mask_flatten shares the same underlying memory as mask
    mask_flatten = mask.view(N, numChannels, -1)

    # Generate the boolean tensor
    zeros = torch.zeros_like(inputTensor, dtype=torch.float)
    booleans = torch.isclose(inputTensor, zeros, atol=threshold)
    booleans_flatten = booleans.view(N, numChannels, -1)
    for c in range(0, numChannels, clusterSize):
        cEnd = min(c + clusterSize, numChannels)
        source = torch.cumprod(booleans_flatten[:, c:cEnd, :], dim=1)[:, -1, :].unsqueeze(1)
        reference = torch.zeros_like(source)
        mask_flatten[:, c:cEnd, :] = torch.isclose(source, reference)

    return mask


class channelClusterGroupLassoPruningMethod(prune.BasePruningMethod):
    """
      Prune according to the specified cluster size along the channel dimension
      Caution: this is an unstructured pruning method, but do not use prune.global_unstructured
      on it.
      Reason: prune.global_unstructured lumps all the tensors in flattened tensor
    """
    PRUNING_TYPE = "unstructured"

    def __init__(self, clusterSize, threshold):
        """
          clusterSize: integer. The number of consecutive elements considered for pruning at once
          threshold: float. How close to zero should be counted as zero.
        """
        super(channelClusterGroupLassoPruningMethod, self).__init__()
        self.threshold = threshold
        self.clusterSize = clusterSize

    def compute_mask(self, t, default_mask):
        """
          t: input tensor
          default_mask: not used
        """
        mask = compute_group_lasso_mask(inputTensor=t, clusterSize=self.clusterSize, threshold=self.threshold)

        return mask

    @classmethod
    def apply(cls, module, name, clusterSize, threshold):
        return super(channelClusterGroupLassoPruningMethod, cls).apply(module, name
                                                                       , clusterSize=clusterSize, threshold=threshold)


###Helper functions for applying pruning####
def applyClusterPruning(module, name, clusterSize, threshold):
    """Prunes tensor corresponding to parameter called `name` in `module`
    by removing every other entry in the tensors.
    Modifies module in place (and also return the modified module)
    by:
    1) adding a named buffer called `name+'_mask'` corresponding to the
    binary mask applied to the parameter `name` by the pruning method.
    The parameter `name` is replaced by its pruned version, while the
    original (unpruned) parameter is stored in a new parameter named
    `name+'_orig'`.

    Args:
        module (nn.Module): module containing the tensor to prune
        name (string): parameter name within `module` on which pruning
                will act.
        clusterSize (int):
        threshold (float):

    Returns:
        module (nn.Module): modified (i.e. pruned) version of the input
            module

    Examples:
        >>> m = nn.Linear(3, 4)
        >>> applyClusterPruning(m, name='bias', clusterSize=3, threshold=1e-3)
    """
    channelClusterGroupLassoPruningMethod.apply(module, name, clusterSize, threshold)
    return module


###Prune a network#####
def pruneNetwork(net, clusterSize, threshold, prefix=''):
    """
    Applies cluster pruning to the 2D-convolution, linear, and 2D-transposed convolution layers
    in a network
    :param net: torch.nn. The network model to be pruned.
    :param clusterSize: int. The cluster granularity size
    :param threshold: float. Values with magnitudes lower than this will be considered as 0
    :param prefix: str. Prefix attached to the module name when printing out which modules are pruned.
    :return: None
    """
    for name, module in net.named_children():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            print("Pruning " + prefix + "." + name + " weight")
            applyClusterPruning(module, "weight", clusterSize, threshold)
        else:
            pruneNetwork(module, clusterSize, threshold, prefix + "." + name)


###Remove pruning masks from a network###
def unPruneNetwork(net):
    """
    Remove the pruning hooks and masks of weights from a pruned network
    :param net: The network to be pruned
    :return: None
    """
    for name, module in net.named_modules():
        for _, hook in module._forward_pre_hooks.items():
            if isinstance(hook, prune.BasePruningMethod):
                prune.remove(module, "weight")
                continue

###Regularization contribution calculation#####
# TODO: Make this run faster
def calculateChannelGroupLasso(input: torch.Tensor, clusterSize=2) -> torch.Tensor:
    """
        Compute the group lasso according to the block size along channels
        input: torch.Tensor. The input tensor
        clusterSize: scalar. Lasso group size
        return: scalar. The group lasso size.
      """
    accumulate = torch.tensor(0, dtype=torch.float32)
    if input.dim() <= 1:
        raise ImportError("Input tensor dimensions must be at least 2")

    numChannels = input.shape[1]
    numChunks = (numChannels - 1) // clusterSize + 1
    eps = 1e-16

    squared = torch.pow(input, 2.0)

    # TODO: The more chunks there are, the slower this gets.... Fix this!
    # Each chunk is a view of the original tensor, so there is no copy overhead
    if clusterSize > 1:
        for chunk in list(torch.chunk(input=squared, chunks=numChunks, dim=1)):
            square_summed = torch.sum(chunk, 1, keepdim=False)
            sqrt = square_summed.add_(torch.tensor(eps)).pow_(0.5)
            accumulate.add_(torch.sum(sqrt))
    elif clusterSize == 1:
        sqrt = squared.add_(torch.tensor(eps)).pow_(0.5)
        accumulate.add_(torch.sum(sqrt))
    return accumulate


def compute_balanced_pruning_mask(
    weightTensor: torch.Tensor,
    clusterSize: int,
    pruneRangeInCluster: int,
    sparsity: float) -> torch.Tensor:
    """
    Calculates the mask for balanced pruning.
    Consecutive weights within the same pruning range can be pruned as clusters
    If the input channel size is not a multiple of prune_range * cluster_size
    then zeros are padded after each input channel during the calculation of masks
    :param weightTensor: The weight tensor to be pruned
    :param clusterSize:  Number of consecutive weights within the same pruning range regarded as one unit
    :param pruneRangeInCluster: Pruning range counted in clusters
    :param sparsity: The target sparsity. Must be greater than 0.0 and smaller than 1.0
    :return:
    """
    if sparsity >= 1.0:
        # Keep the sparsity strictly less than 1.0,
        # Otherwise, all the weights might become zero
        sparsity = 0.9999

    if sparsity >= 1.0 / pruneRangeInCluster:
        inputDims = weightTensor.size()
        numInputChannels = inputDims[1]
        # Number of filters. Each filter is of dimensions CHW
        N = inputDims[0]
        # Prune range in terms of individual weights
        pruneRangeInWeights = pruneRangeInCluster * clusterSize

        # Lower the weight matrix into a 3D tensor of dimensions N x C x (HW)
        weightTensorFlatten = weightTensor.view(N, numInputChannels, -1)
        # Need to permute the flattened weight tensor to get HHWC layout
        # GOTTCHA: Need to call contiguous, otherwise view will not work
        # See https://discuss.pytorch.org/t/call-contiguous-after-every-permute-call/13190
        weightTensorNHWxC = (weightTensorFlatten.permute(0, 2, 1).contiguous()).view(-1, numInputChannels)
        numWeightRows = weightTensorNHWxC.size()[0]
        # Allocate the mask tensor in NHWC layout
        maskNHWxC = torch.zeros_like(weightTensorNHWxC)

        # consider each matrix row on a chunk-by-chunk basis
        for row in range(0, numWeightRows):
            # Extract each row from the flattened weight matrix
            # and pad it with zeros to a multiple of clusterSize * pruneRange
            paddedLen = (1 + (numInputChannels - 1) // pruneRangeInWeights) * pruneRangeInWeights
            weightRowWithPadding = torch.zeros(paddedLen, dtype=torch.float)
            weightRowWithPadding[0:numInputChannels] = torch.squeeze(weightTensorNHWxC[row, :])
            # Split each row into chunks. Each chunk is a prune range
            weightRowWithPaddingPartitioned = weightRowWithPadding.view(-1, pruneRangeInCluster, clusterSize)
            # Calculate the norm of each chunk.
            # Dim 0: Across ranges
            # Dim 1: Across clusters within the same range
            # Dim 2: Across values within the same cluster
            # TODO: Currently using L_inf. Change the norm order if necessary
            norms = torch.norm(weightRowWithPaddingPartitioned, p=float(1.0), dim=2, keepdim=True).detach().numpy()
            threshold = np.quantile(norms, q=sparsity, axis=1, interpolation='lower', keepdims=True)
            # print ("Row: {}, threshold: {}".format(row, threshold))
            # Generate the padded mask and flatten it
            rowMaskPadded = torch.from_numpy(np.greater(norms, threshold).astype(float)).flatten().repeat_interleave(clusterSize)
            maskNHWxC[row, :] = rowMaskPadded[0:numInputChannels]

        mask = (maskNHWxC.view(N, -1, numInputChannels).permute(0, 2,1).contiguous()).view_as(weightTensor)
    else:
        # Special case: if the sparsity is lower than
        mask = torch.ones_like(weightTensor)
    # print('Mask: {}'.format(mask))
    return mask


class balancedPruningMethod(prune.BasePruningMethod):
    """
      Perform balanced pruning according to the specified cluster size along the channel dimension
      Caution: this is an unstructured pruning method, but do not use prune.global_unstructured
      on it.
      Reason: prune.global_unstructured lumps all the tensors in flattened tensor
    """
    PRUNING_TYPE = "unstructured"

    def __init__(self, clusterSize: int, pruneRangeInCluster: int, sparsity: float):
        """
          clusterSize: integer. The number of consecutive elements considered for pruning at once
          sparsity: float. Target sparsity
        """
        super(balancedPruningMethod, self).__init__()
        self.pruneRangeInCluster = pruneRangeInCluster
        self.clusterSize = clusterSize
        self.sparsity = sparsity

    def compute_mask(self, t, default_mask):
        """
          t: input tensor
          default_mask: not used
        """
        mask = compute_balanced_pruning_mask(
            weightTensor=t,
            clusterSize=self.clusterSize,
            pruneRangeInCluster=self.pruneRangeInCluster,
            sparsity=self.sparsity)

        return mask

    @classmethod
    def apply(cls, module, name, clusterSize, pruneRangeInCluster, sparsity):
        return super(balancedPruningMethod, cls).apply(module, name,
                                                       clusterSize=clusterSize,
                                                       pruneRangeInCluster=pruneRangeInCluster,
                                                       sparsity=sparsity)


###Helper functions for applying balanced pruning####
def applyBalancedPruning(module, name, clusterSize: int, pruneRangeInCluster: int, sparsity: float):
    """Prunes tensor corresponding to parameter called `name` in `module`
    by removing every other entry in the tensors.
    Modifies module in place (and also return the modified module)
    by:
    1) adding a named buffer called `name+'_mask'` corresponding to the
    binary mask applied to the parameter `name` by the pruning method.
    The parameter `name` is replaced by its pruned version, while the
    original (unpruned) parameter is stored in a new parameter named
    `name+'_orig'`.

    Args:
        module (nn.Module): module containing the tensor to prune
        name (string): parameter name within `module` on which pruning
                will act.
        clusterSize (int): Number of consecutive weights to be seen as one unit
        pruneRangeInCluster (int): Size of balanced pruning window in terms of cluster
        sparsity (float): Target sparsity level
    Side-effects:
        - Adds pruning attributes to the module
        - Adds a registered buffer called weight_target_sparsity to the module
    Returns:
        module (nn.Module): modified (i.e. pruned) version of the input
            module

    Examples:
        >>> m = nn.Linear(3, 4)
        >>> applyBalancedPruning(m, name='weight', clusterSize=3, sparsity=0.25, pruneRangeInCluster=4)
    """
    balancedPruningMethod.apply(module, name, clusterSize, pruneRangeInCluster, sparsity)
    module.register_buffer('weight_target_sparsity', torch.tensor(float(sparsity)))
    return module

def savePruneMask(net) -> None:
    """
    Saves all the masks as module attributes.
    Useful for temprorarily saving the masks before quantization
    Side-effect: add attributes "[tensor_name]_mask" to the module
    :param net:
    :return:
    """
    for name, module in net.named_modules():
        for _, hook in module._forward_pre_hooks.items():
            if isinstance(hook, prune.BasePruningMethod):
                setattr(module, hook._tensor_name+'_prev_prune_mask', module.weight_mask.detach().clone())
                continue

def restoreWeightPruneMask(net) -> None:
    """
    Reapply weight masks from saved values
    Side-affects:
    - Reinstate the saved weight masks as a pruning mask
    - Delete the saved weight mask from the modulle
    :param net:
    :return:
    """
    for name, module in net.named_modules():
        if hasattr(module, 'weight_prev_prune_mask'):
            prune.CustomFromMask.apply(module, 'weight', module.weight_prev_prune_mask)
            del module.weight_prev_prune_mask
            continue
