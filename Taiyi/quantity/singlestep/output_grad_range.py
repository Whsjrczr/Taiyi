import torch

from .base_class import SingleStepQuantity
from ...extensions import BackwardOutputExtension


class OutputGradRange(SingleStepQuantity):

    def _compute(self, global_step):
        grad = getattr(self._module, "output_grad", None)
        if grad is None:
            return None
        if not torch.isfinite(grad).all():
            return None
        return {
            "min": grad.min(),
            "max": grad.max(),
            "abs_max": grad.abs().max(),
        }

    def backward_extensions(self):
        return [BackwardOutputExtension()]
