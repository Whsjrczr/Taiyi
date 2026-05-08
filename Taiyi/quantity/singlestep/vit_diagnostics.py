import torch

from .base_class import SingleStepQuantity
from ...extensions import ForwardInputExtension, ForwardOutputExtension


def _float_detached(tensor):
    return tensor.detach().float()


def _std(tensor, dim=None):
    return tensor.std(dim=dim, unbiased=False)


def _var(tensor):
    return tensor.var(unbiased=False)


def _rms(tensor):
    x = _float_detached(tensor)
    return torch.sqrt(torch.mean(x * x))


def _safe_ratio(numerator, denominator):
    numerator = float(numerator.item() if isinstance(numerator, torch.Tensor) else numerator)
    denominator = float(denominator.item() if isinstance(denominator, torch.Tensor) else denominator)
    if denominator == 0.0:
        return float("nan")
    return numerator / denominator


def _summary(prefix, tensor):
    x = _float_detached(tensor)
    return {
        f"{prefix}_mean": x.mean(),
        f"{prefix}_std": _std(x),
        f"{prefix}_min": x.min(),
        f"{prefix}_max": x.max(),
        f"{prefix}_abs_max": x.abs().max(),
    }


def _tensor_stats(prefix, tensor):
    x = _float_detached(tensor)
    return {
        f"{prefix}_mean": x.mean(),
        f"{prefix}_std": _std(x),
        f"{prefix}_var": _var(x),
        f"{prefix}_rms": _rms(x),
        f"{prefix}_min": x.min(),
        f"{prefix}_max": x.max(),
        f"{prefix}_abs_max": x.abs().max(),
        f"{prefix}_nan_count": torch.isnan(x).sum(),
        f"{prefix}_inf_count": torch.isinf(x).sum(),
    }


def _axis_stats(prefix, tensor):
    x = _float_detached(tensor)
    if x.dim() != 3:
        return {}

    stats = {}
    axis_values = {
        "across_channel_mean": x.mean(dim=-1),
        "across_channel_std": _std(x, dim=-1),
        "across_sequence_mean": x.mean(dim=1),
        "across_sequence_std": _std(x, dim=1),
        "token_batch_channel_mean": x.mean(dim=(0, 2)),
        "token_batch_channel_std": _std(x, dim=(0, 2)),
    }
    for axis_name, value in axis_values.items():
        stats.update(_summary(f"{prefix}_{axis_name}", value))
    return stats


def _cls_patch_stats(prefix, tensor):
    x = _float_detached(tensor)
    if x.dim() != 3 or x.shape[1] < 2:
        return {}

    cls = x[:, 0, :]
    patch = x[:, 1:, :]
    cls_rms = _rms(cls)
    patch_rms = _rms(patch)
    return {
        f"{prefix}_cls_rms": cls_rms,
        f"{prefix}_patch_rms": patch_rms,
        f"{prefix}_cls_patch_rms_ratio": _safe_ratio(cls_rms, patch_rms),
        f"{prefix}_cls_mean": cls.mean(),
        f"{prefix}_patch_mean": patch.mean(),
    }


def _running_stats(module):
    stats = {}
    for child_name, child in module.named_modules():
        suffix = "self" if child_name == "" else child_name.replace(".", "_")
        running_mean = getattr(child, "running_mean", None)
        running_var = getattr(child, "running_var", None)
        if running_mean is not None:
            x = _float_detached(running_mean)
            stats.update(
                {
                    f"{suffix}_running_mean_mean": x.mean(),
                    f"{suffix}_running_mean_std": _std(x),
                    f"{suffix}_running_mean_abs_max": x.abs().max(),
                }
            )
        if running_var is not None:
            x = _float_detached(running_var)
            stats.update(
                {
                    f"{suffix}_running_var_mean": x.mean(),
                    f"{suffix}_running_var_std": _std(x),
                    f"{suffix}_running_var_min": x.min(),
                    f"{suffix}_running_var_max": x.max(),
                }
            )
    return stats


class ViTNormStats(SingleStepQuantity):
    def _compute(self, global_step):
        stats = {}
        for prefix, tensor in (("input", self._module.input), ("output", self._module.output)):
            stats.update(_tensor_stats(prefix, tensor))
            stats.update(_axis_stats(prefix, tensor))
            stats.update(_cls_patch_stats(prefix, tensor))
        stats.update(_running_stats(self._module))
        return stats

    def forward_extensions(self):
        return [ForwardInputExtension(), ForwardOutputExtension()]


class ViTResidualStats(SingleStepQuantity):
    def _compute(self, global_step):
        states = getattr(self._module, "residual_states", None)
        if not states:
            return None
        stats = {}
        for branch_name in ("attn", "mlp"):
            state = states.get(branch_name, {})
            stream = state.get("stream")
            branch = state.get("branch")
            if stream is None or branch is None:
                continue
            stream_rms = _rms(stream)
            branch_rms = _rms(branch)
            stats[f"{branch_name}_stream_rms"] = stream_rms
            stats[f"{branch_name}_branch_rms"] = branch_rms
            stats[f"{branch_name}_branch_to_stream_ratio"] = _safe_ratio(branch_rms, stream_rms)
        return stats


class ViTLogitsStats(SingleStepQuantity):
    def _compute(self, global_step):
        logits = _float_detached(self._module.output)
        probs = torch.softmax(logits, dim=-1)
        entropy = -(probs * probs.clamp_min(torch.finfo(probs.dtype).tiny).log()).sum(dim=-1)
        return {
            "logits_mean": logits.mean(),
            "logits_std": _std(logits),
            "logits_rms": _rms(logits),
            "logits_abs_max": logits.abs().max(),
            "softmax_entropy_mean": entropy.mean(),
            "softmax_max_prob_mean": probs.max(dim=-1).values.mean(),
        }

    def forward_extensions(self):
        return [ForwardOutputExtension()]
