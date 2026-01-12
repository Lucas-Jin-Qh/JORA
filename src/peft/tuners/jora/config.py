from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Union, List

from peft.config import PeftConfig
from peft.utils import PeftType

RotationParam = Literal["cayley", "angle"]
RotationImpl = Literal["auto", "triton", "torch"]
CoreType = Literal["diag", "block", "lowrank"]
SelectionType = Literal["topk_ema", "random", "none"]
MagnitudeType = Literal["none", "ecd_tanh", "oer_softmax"]

@dataclass
class JoraConfig(PeftConfig):
    """Configuration for the JORA tuner.

    This class defines all hyperparameters for JORA (Joint Orthogonal Rotation Adaptation).
    The implementation maintains compatibility with legacy JORA while providing
    cleaner parameter organization.
    """

    # Required by PEFT mapping
    peft_type: Optional[PeftType] = field(default=PeftType.JORA, init=False)

    # ---- PEFT common knobs ----
    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={"help": "Names (or regex) of modules to replace, e.g. ['q_proj','v_proj']."},
    )
    exclude_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={"help": "Module names/regex to exclude from adaptation."},
    )
    modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={"help": "Additional modules to keep trainable & save with adapter."},
    )
    target_parameters: Optional[list[str]] = field(
        default=None,
        metadata={
            "help": (
                "List of parameter names or regex expression of the parameter names to replace with JORA. "
                "This argument behaves similarly to `target_modules`, except that the parameter name should be passed. "
                "Generally, you should use `target_modules` to target the module (e.g. `nn.Linear`). However, in some "
                "circumstances, this is not possible. E.g., in many mixture of expert (MoE) layers in HF Transformers, "
                "instead of using `nn.Parameter`, an `nn.Parameter` is used. PEFT normally overwrites the `forward` "
                "method for JORA, but for `nn.Parameter`, there is none. Therefore, to apply JORA to that parameter, "
                "it needs to be targeted with `target_parameters`."
            )
        },
    )
    inference_mode: bool = field(default=False)

    # ---- JORA: rotation ----
    S_L: int = 32
    S_R: int = 32
    k: int = 8

    rotation_param: RotationParam = "cayley"
    rotation_impl: RotationImpl = "auto"
    max_angle: float = 0.1

    # If you want a strict no-rotation ablation, set S_L=0 and/or S_R=0.
    force_random_rotation_init: bool = True
    theta_init_std: float = 0.02

    # Performance optimization flags
    use_parallel_rotations: bool = False
    use_fast_ecd: bool = False
    use_optimized_pair_selection: bool = True

    # Learning rate parameters for controlling theta freezing
    lr_theta: float = 0.05  # Theta learning rate (set to 0.0 to freeze theta for fast mode)
    lr_core: float = 0.01   # Core learning rate

    # ---- JORA: pair selection / schedule ----
    selection: SelectionType = "topk_ema"
    ema_beta: float = 0.98
    warmup_steps: int = 0  # 0 disables warmup gating
    warmup_ratio: float = 0.0  # if >0 and total_steps is provided, warmup_steps = int(total_steps*warmup_ratio)
    update_interval: int = 1  # update selection every N forward steps
    ema_update_interval: int = 1  # update EMA statistics every N forward steps
    use_gumbel: bool = False
    gumbel_tau: float = 1.0

    # Performance optimization: group modules for shared selection
    selection_group_size: int = 1  # 1 = per-module, >1 = grouped selection, 0 = global shared
    selection_group_by: str = "dimension"  # "dimension", "type", "none"

    # ---- JORA: core ----
    core: CoreType = "diag"
    zero_init_core: bool = False
    block_size: int = 4
    lowrank_r: int = 8
    lowrank_alpha: Optional[float] = None

    # ---- JORA: magnitude module ----
    magnitude: MagnitudeType = "oer_softmax"

    # Legacy ECD compatibility (automatically sets magnitude to "ecd_tanh" if enabled)
    use_ecd: bool = False
    ecd_temperature: float = 1.0  # used for legacy ecd_tanh
    oer_temperature: float = 1.0  # used for oer_softmax
    ecd_alpha: float = 0.5
    ecd_learnable_energy: bool = False

    ecd_temp_annealing: bool = False
    ecd_temp_start: float = 5.0
    ecd_temp_end: float = 1.0

    # ---- JORA: magnitude extra knobs (optional) ----
    magnitude_init_scale: float = 1.0
    magnitude_lr_scale: float = 1.0
    chunk_size: int = 512  # for OER, if enabled

    # ---- misc ----
    single_sided: Literal["none", "left", "right"] = "none"
    eps: float = 1e-8

    def __post_init__(self):
        # Call parent __post_init__ if it exists (compat with different PEFT versions)
        parent_post = getattr(super(), "__post_init__", None)
        if callable(parent_post):
            parent_post()

        # Backward-compat for legacy flags
        # Legacy ECD flags map onto the modular magnitude selector.
        if getattr(self, "ablate_ecd", False):
            self.magnitude = "none"
        elif getattr(self, "use_ecd", False):
            self.magnitude = "ecd_tanh"

        # Backward-compat for legacy flags
        if getattr(self, "use_triton", None) is True and self.rotation_impl == "auto":
            self.rotation_impl = "triton"
        if getattr(self, "use_cayley", None) is True:
            self.rotation_param = "cayley"
        if getattr(self, "use_cayley", None) is False:
            self.rotation_param = "angle"

    # ---- distributed / DDP ----
    ddp_allow_unused_parameters: bool = True  # sparse selections may leave some params unused per step
