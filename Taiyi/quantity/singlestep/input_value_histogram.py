import torch

from .base_class import SingleStepQuantity
from ...extensions import ForwardInputExtension


class InputValueHistogram(SingleStepQuantity):
    """Histogram summary of flattened per-sample input values."""

    def __init__(self, module, track_schedule, num_bins=64):
        super().__init__(module, track_schedule)
        self.num_bins = int(num_bins)

    def _compute(self, global_step):
        data = self._module.input
        if data is None:
            return None

        flat = data.detach().reshape(data.shape[0], -1)
        if flat.numel() == 0:
            return None

        values = flat.reshape(-1)
        abs_values = values.abs()
        value_hist, value_edges = torch.histogram(values, bins=self.num_bins)
        abs_hist, abs_edges = torch.histogram(abs_values, bins=self.num_bins)

        return {
            "num_samples": int(flat.shape[0]),
            "num_dims_per_sample": int(flat.shape[1]),
            "value_hist": value_hist,
            "value_bin_edges": value_edges,
            "abs_value_hist": abs_hist,
            "abs_value_bin_edges": abs_edges,
            "value_mean": values.mean(),
            "value_std": values.std(unbiased=False),
            "abs_value_mean": abs_values.mean(),
            "abs_value_std": abs_values.std(unbiased=False),
        }

    def forward_extensions(self):
        return [ForwardInputExtension()]
