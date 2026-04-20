import math

import torch

from .base_class import SingleStepQuantity
from ...extensions import ForwardInputExtension


class InputNormContribution(SingleStepQuantity):
    """Summarize how LN/RMS normalization energy is shared across dims."""

    def _compute(self, global_step):
        data = self._module.input
        if data is None:
            return None

        flat = data.detach().reshape(data.shape[0], -1)
        if flat.numel() == 0:
            return None

        return {
            "num_samples": int(flat.shape[0]),
            "num_dims_per_sample": int(flat.shape[1]),
            "topk": int(min(5, flat.shape[1])),
            "rms": self._summarize_share(flat.pow(2)),
            "ln": self._summarize_share((flat - flat.mean(dim=1, keepdim=True)).pow(2)),
        }

    def forward_extensions(self):
        return [ForwardInputExtension()]

    def _summarize_share(self, energies):
        denom = energies.sum(dim=1, keepdim=True)
        valid = denom.squeeze(1) > 0
        if not torch.any(valid):
            return {
                "valid_samples": 0,
                "top1_share_mean": float("nan"),
                "top1_share_std": float("nan"),
                "topk_share_mean": float("nan"),
                "topk_share_std": float("nan"),
                "effective_dims_mean": float("nan"),
                "effective_dims_std": float("nan"),
                "normalized_entropy_mean": float("nan"),
                "normalized_entropy_std": float("nan"),
            }

        share = energies[valid] / denom[valid]
        sorted_share, _ = torch.sort(share, dim=1, descending=True)
        topk = min(5, share.shape[1])
        top1 = sorted_share[:, 0]
        topk_share = sorted_share[:, :topk].sum(dim=1)
        effective_dims = 1.0 / share.pow(2).sum(dim=1)
        entropy = -(share * torch.log(share.clamp_min(1e-12))).sum(dim=1)
        if share.shape[1] > 1:
            normalized_entropy = entropy / math.log(share.shape[1])
        else:
            normalized_entropy = torch.ones_like(entropy)

        return {
            "valid_samples": int(share.shape[0]),
            "top1_share_mean": top1.mean(),
            "top1_share_std": top1.std(unbiased=False),
            "topk_share_mean": topk_share.mean(),
            "topk_share_std": topk_share.std(unbiased=False),
            "effective_dims_mean": effective_dims.mean(),
            "effective_dims_std": effective_dims.std(unbiased=False),
            "normalized_entropy_mean": normalized_entropy.mean(),
            "normalized_entropy_std": normalized_entropy.std(unbiased=False),
        }
