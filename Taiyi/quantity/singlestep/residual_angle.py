from .base_class import SingleStepQuantity

import torch


def _flatten_batch(data):
    if data.dim() == 0:
        return data.reshape(1, 1)
    if data.dim() == 1:
        return data.reshape(1, -1)
    return data.reshape(data.shape[0], -1)


def _pairwise_angle(lhs, rhs):
    lhs = _flatten_batch(lhs.float())
    rhs = _flatten_batch(rhs.float())
    dot = (lhs * rhs).sum(dim=1)
    denom = lhs.norm(p=2, dim=1) * rhs.norm(p=2, dim=1)
    cos = dot / denom.clamp_min(1e-12)
    cos = cos.clamp(-1.0, 1.0)
    return torch.acos(cos) * (180.0 / torch.pi)


class _ResidualAngleBase(SingleStepQuantity):
    source_key = None
    target_key = None
    reduce = "mean"

    def _compute(self, global_step):
        residual_states = getattr(self._module, "residual_states", None)
        if not residual_states:
            return None

        result = {}
        for state_name, state in residual_states.items():
            if self.source_key not in state or self.target_key not in state:
                continue
            theta = _pairwise_angle(state[self.source_key], state[self.target_key])
            if self.reduce == "mean":
                result[state_name] = theta.mean()
            else:
                result[state_name] = theta.std()

        if len(result) == 0:
            return None
        return result


class ResidualInputAngleMean(_ResidualAngleBase):
    source_key = "stream"
    target_key = "branch"
    reduce = "mean"


class ResidualInputAngleStd(_ResidualAngleBase):
    source_key = "stream"
    target_key = "branch"
    reduce = "std"


class ResidualStreamOutputAngleMean(_ResidualAngleBase):
    source_key = "stream"
    target_key = "output"
    reduce = "mean"


class ResidualStreamOutputAngleStd(_ResidualAngleBase):
    source_key = "stream"
    target_key = "output"
    reduce = "std"
