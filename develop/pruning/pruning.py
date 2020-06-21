import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

import custom_modules.custom_modules as modules


def compute_mask(inputTensor : torch.Tensor, clusterSize : int, threshold : float) -> torch.Tensor:
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
        mask = compute_mask(inputTensor=t, clusterSize=self.clusterSize, threshold=self.threshold)

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
    Remove the pruning hooks and masks from a pruned network
    :param net: The network to be pruned
    :return: None
    """
    for name, module in net.named_modules():
        for _, hook in module._forward_pre_hooks.items():
            if isinstance(hook, prune.BasePruningMethod):
                prune.remove(module, "weight")
                continue


###Regularization contribution calculation#####
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

    for chunk in list(torch.chunk(input=input, chunks=numChunks, dim=1)):
        squared = torch.pow(chunk, 2.0)
        square_summed = torch.sum(squared, 1, keepdim=False)
        sqrt = torch.pow(square_summed.add_(torch.tensor(eps)), 0.5)
        accumulate.add_(torch.sum(sqrt))
    return accumulate
