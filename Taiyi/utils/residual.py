from collections import defaultdict

import torch


REQUIRED_KEYS = ("stream", "branch", "output")


def _module_name_matches_residual(module_name, module):
    text = f"{module_name}.{module.__class__.__name__}".lower()
    keywords = ("residual", "resblock", "basicblock", "bottleneck", "block")
    return any(keyword in text for keyword in keywords)


def _validate_residual_states(residual_states):
    if not isinstance(residual_states, dict) or len(residual_states) == 0:
        return False, "residual_states must be a non-empty dict"

    details = {}
    valid = True
    for state_name, state in residual_states.items():
        if not isinstance(state, dict):
            valid = False
            details[state_name] = "state must be a dict"
            continue

        missing = [key for key in REQUIRED_KEYS if key not in state]
        if missing:
            valid = False
            details[state_name] = f"missing keys: {missing}"
            continue

        tensor_errors = []
        batch_size = None
        for key in REQUIRED_KEYS:
            value = state[key]
            if not isinstance(value, torch.Tensor):
                tensor_errors.append(f"{key} is not a tensor")
                continue
            if batch_size is None:
                batch_size = value.shape[0] if value.dim() > 0 else 1
            else:
                current_batch = value.shape[0] if value.dim() > 0 else 1
                if current_batch != batch_size:
                    tensor_errors.append(f"{key} batch size mismatch")

        if tensor_errors:
            valid = False
            details[state_name] = "; ".join(tensor_errors)
        else:
            details[state_name] = "ok"

    return valid, details


@torch.no_grad()
def check_residual_compatibility(model, sample_inputs=None, sample_kwargs=None):
    """
    Inspect modules for Taiyi residual-angle compatibility.

    Args:
        model: nn.Module to inspect.
        sample_inputs: optional tensor / tuple / list for running one forward pass.
        sample_kwargs: optional kwargs for the forward pass.

    Returns:
        dict with compatible / incompatible / possible_residual_modules / summary.
    """
    sample_kwargs = sample_kwargs or {}

    if sample_inputs is not None:
        was_training = model.training
        model.eval()
        if isinstance(sample_inputs, (tuple, list)):
            model(*sample_inputs, **sample_kwargs)
        else:
            model(sample_inputs, **sample_kwargs)
        model.train(was_training)

    compatible = {}
    incompatible = {}
    possible_residual_modules = []

    for module_name, module in model.named_modules():
        if module_name == "":
            continue

        has_states = hasattr(module, "residual_states")
        looks_residual = _module_name_matches_residual(module_name, module)
        if looks_residual:
            possible_residual_modules.append(module_name)

        if not has_states:
            if looks_residual:
                incompatible[module_name] = "missing residual_states"
            continue

        ok, details = _validate_residual_states(module.residual_states)
        if ok:
            compatible[module_name] = details
        else:
            incompatible[module_name] = details

    return {
        "compatible": compatible,
        "incompatible": incompatible,
        "possible_residual_modules": possible_residual_modules,
        "summary": {
            "compatible_count": len(compatible),
            "incompatible_count": len(incompatible),
            "possible_residual_count": len(possible_residual_modules),
            "forward_executed": sample_inputs is not None,
        },
    }
