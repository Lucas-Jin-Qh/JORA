from __future__ import annotations

from typing import Any, Optional, Tuple

import torch
import torch.nn as nn

def is_conv1d_layer(module: nn.Module) -> bool:
    # HF Conv1D used in GPT-2 family
    return module.__class__.__name__ == "Conv1D"

def get_in_out_features(module: nn.Module) -> Tuple[int, int]:
    if isinstance(module, nn.Linear):
        return int(module.in_features), int(module.out_features)
    if is_conv1d_layer(module):
        # HF Conv1D: weight shape [in, out] (opposite of Linear)
        w = module.weight
        return int(w.shape[0]), int(w.shape[1])
    raise TypeError(f"Unsupported module type for JORA: {type(module)}")

def linear_forward(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if isinstance(module, nn.Linear):
        return nn.functional.linear(x, module.weight, module.bias)
    if is_conv1d_layer(module):
        # Conv1D forward from transformers.pytorch_utils.Conv1D
        size_out = x.size()[:-1] + (module.weight.shape[1],)
        x_flat = x.view(-1, x.size(-1))
        out = torch.matmul(x_flat, module.weight)  # weight [in, out]
        if module.bias is not None:
            out = out + module.bias
        return out.view(size_out)
    raise TypeError(f"Unsupported module type for JORA: {type(module)}")
