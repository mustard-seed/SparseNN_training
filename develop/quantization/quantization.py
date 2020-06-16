from torch.quantization.observer import *
from torch.quantization import QuantStub, DeQuantStub
import torch                                        # root package
import torch.nn as nn                               # neural networks


# Helper function used to round a given tensor to integer power of 2
def round_to_power_of_two(t):
  """
    t (torch.tensor):  Then tensor to be rounded

    Return (torch.tensor): The rounded tensor
  """
  log2 = torch.log(torch.full_like(t, 2))
  logBase2T = torch.div(torch.log(t), log2)
  logBase2TCeil = torch.ceil(logBase2T)
  exponent = log2 * logBase2TCeil
  roundedT = torch.exp(exponent)

  return roundedT


class RoundedHistogramObserver(HistogramObserver):
  """
    Similar to torch.quantizaiton.observer.HistogramObserver
    Difference:
    Differences:
      - Force it qint8, per_tensor_symmetric quantization
      - Quantization scale is rounded to an integer power of 2
  """

  def __init__(self, bins=2048, upsample_rate=128):
    super(RoundedHistogramObserver, self).__init__(
      bins=bins,
      upsample_rate=upsample_rate,
      dtype=torch.qint8,
      qscheme=torch.per_tensor_symmetric
    )

  @torch.jit.export
  def calculate_qparams(self):
    scale, zeroPoint = super(RoundedHistogramObserver, self).calculate_qparams()
    device = 'cuda' if scale.is_cuda else 'cpu'
    # Round the scale
    roundedScale = round_to_power_of_two(scale)
    return roundedScale, zeroPoint


class RoundedMinMaxObserver(MinMaxObserver):
  """
    Similar to torch.quantization.observer.MinMaxObserver
    Differences:
      - Force it qint8, per_tensor_symmetric quantization
      - Quantization scale is rounded to an integer power of 2
  """

  def __init__(self):
    super(RoundedMinMaxObserver, self).__init__(
      dtype=torch.qint8,
      qscheme=torch.per_tensor_symmetric
    )

  @torch.jit.export
  def calculate_qparams(self):
    scale, zeroPoint = super(RoundedMinMaxObserver, self).calculate_qparams()
    device = 'cuda' if scale.is_cuda else 'cpu'
    # Round the scale
    roundedScale = round_to_power_of_two(scale)
    return roundedScale, zeroPoint


class RoundedMovingAverageMinMaxObserver(MovingAverageMinMaxObserver):
  """
    Similar to torch.quantization.observer.MovingAverageObserver
    Differences:
      - Force it qint8, per_tensor_symmetric quantization
      - Quantization scale is rounded to an integer power of 2
  """

  def __init__(self, averaging_constant=0.01):
    super(RoundedMovingAverageMinMaxObserver, self).__init__(
      averaging_constant=averaging_constant,
      dtype=torch.qint8,
      qscheme=torch.per_tensor_symmetric
    )

  @torch.jit.export
  def calculate_qparams(self):
    scale, zeroPoint = super(RoundedMovingAverageMinMaxObserver, self).calculate_qparams()
    device = 'cuda' if scale.is_cuda else 'cpu'
    # Round the scale
    roundedScale = round_to_power_of_two(scale)
    return roundedScale, zeroPoint