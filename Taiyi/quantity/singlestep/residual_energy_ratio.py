from .base_class import SingleStepQuantity

import torch


def _flatten_batch(data):
    if data.dim() == 0:
        return data.reshape(1, 1)
    if data.dim() == 1:
        return data.reshape(1, -1)
    return data.reshape(data.shape[0], -1)


class ResidualEnergyRatio(SingleStepQuantity):
    """
    Residual branch energy ratio:
        ||branch||^2 / (||stream||^2 + ||branch||^2)
    computed sample-wise and averaged over the batch.
    """

    def _compute(self, global_step):
        residual_states = getattr(self._module, "residual_states", None)
        if not residual_states:
            return None

        result = {}
        for state_name, state in residual_states.items():
            if "stream" not in state or "branch" not in state:
                continue

            stream = _flatten_batch(state["stream"].float())
            branch = _flatten_batch(state["branch"].float())
            stream_energy = (stream * stream).sum(dim=1)
            branch_energy = (branch * branch).sum(dim=1)
            total_energy = stream_energy + branch_energy
            ratio = branch_energy / total_energy.clamp_min(1e-12)
            result[state_name] = ratio.mean()

        if len(result) == 0:
            return None
        return result
