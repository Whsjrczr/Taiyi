from .base_class import SingleStepQuantity


class WeightGradRange(SingleStepQuantity):

    def _compute(self, global_step):
        weight = getattr(self._module, "weight", None)
        grad = None if weight is None else weight.grad
        if grad is None:
            return None
        return {
            "min": grad.min(),
            "max": grad.max(),
            "abs_max": grad.abs().max(),
        }
