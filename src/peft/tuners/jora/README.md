# JORA Tuner for PEFT

This module implements the JORA (Joint Orthogonal Rotation Adaptation) tuner for the PEFT library.

## Overview

JORA is a parameter-efficient fine-tuning method that combines orthogonal rotations with sparse parameter selection to achieve better adaptation with fewer trainable parameters.

## Architecture

```
peft/tuners/jora/
├── __init__.py      # Module exports and PEFT registration
├── config.py        # JoraConfig class with hyperparameters
├── model.py         # JoraModel class (BaseTuner implementation)
├── layer.py         # JoraLayer class (BaseTunerLayer implementation)
├── core.py          # Core adaptation matrices
├── rotation.py      # Rotation parameter handling
├── selection.py     # Parameter selection algorithms
├── magnitude.py     # Magnitude scaling functions
└── utils.py         # Utility functions
```

## Features Implemented

- **Layer Wrapper**: JoraLayer for nn.Linear and HF Conv1D layers
- **Parameter Selection**: Top-k pair selection with warmup scheduling
- **Rotation**: Torch-based autograd-safe rotation implementation
- **Core Types**: DiagCore, BlockCore, LowRankCore
- **Magnitude Scaling**: ECD scaling matching legacy implementation
- **Sparse Selection**: EMA-based parameter selection for efficiency

## Quick Start

```python
from peft import get_peft_model, JoraConfig

# Configure JORA
config = JoraConfig(
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    S_L=32,  # Left rotation dimension
    S_R=32,  # Right rotation dimension
    k=8,     # Selection parameter
)

# Apply to model
model = get_peft_model(model, config)
```

## API Reference

### JoraConfig

Configuration class for JORA hyperparameters.

#### Constructor Parameters

**PEFT Common Parameters:**
- `target_modules: Optional[Union[List[str], str]]` - Names of modules to adapt
- `exclude_modules: Optional[Union[List[str], str]]` - Modules to exclude from adaptation
- `modules_to_save: Optional[List[str]]` - Additional modules to keep trainable
- `inference_mode: bool` - Whether adapter is frozen (default: False)

**JORA Rotation Parameters:**
- `S_L: int` - Left rotation matrix dimension (default: 32)
- `S_R: int` - Right rotation matrix dimension (default: 32)
- `rotation_param: RotationParam` - Rotation parameterization: "cayley" or "angle" (default: "cayley")
- `rotation_impl: RotationImpl` - Rotation implementation: "auto", "triton", or "torch" (default: "auto")

**JORA Selection Parameters:**
- `k: int` - Number of selected parameter pairs (default: 8)
- `selection: SelectionType` - Selection method: "topk_ema", "random", or "none" (default: "topk_ema")
- `ema_beta: float` - EMA decay factor for selection (default: 0.98)
- `warmup_steps: int` - Warmup steps for selection (default: 0)
- `warmup_ratio: float` - Warmup ratio relative to total steps (default: 0.0)

**JORA Core Parameters:**
- `core: CoreType` - Core adaptation type: "diag", "block", or "lowrank" (default: "diag")
- `zero_init_core: bool` - Initialize core with zeros (default: False)

**JORA Magnitude Parameters:**
- `magnitude: MagnitudeType` - Magnitude scaling: "ecd_tanh", "oer_softmax", or "none" (default: "oer_softmax")
- `ecd_alpha: float` - ECD scaling alpha parameter (default: 0.5)
- `oer_temperature: float` - OER softmax temperature (default: 1.0)

#### Methods

##### `__post_init__()`
Post-initialization hook that sets up legacy compatibility flags.

### JoraModel

PEFT BaseTuner implementation for JORA.

#### Constructor Parameters

- `model: nn.Module` - Base model to adapt
- `peft_config: JoraConfig` - JORA configuration
- `adapter_name: str` - Adapter name (default: "default")

#### Methods

##### `set_total_steps(total_steps: int)`
Set the total number of training steps for warmup scheduling.

**Parameters:**
- `total_steps: int` - Total training steps

##### `jora_update_step()`
Manually trigger parameter selection update (usually called automatically via hooks).

##### `jora_update_temperature(current_step: int, total_steps: int)`
Manually update temperature annealing (usually called automatically via hooks).

**Parameters:**
- `current_step: int` - Current training step
- `total_steps: int` - Total training steps

##### `set_adapter(adapter_name: str | list[str], inference_mode: bool = False)`
Set active adapter(s).

**Parameters:**
- `adapter_name: str | list[str]` - Adapter name(s) to activate
- `inference_mode: bool` - Whether to freeze adapter parameters

##### `disable_adapter_layers()`
Disable all JORA adapter layers (set to identity).

##### `enable_adapter_layers()`
Enable all JORA adapter layers.

##### `prepare_inputs_for_generation(*args, **kwargs)`
Delegate input preparation to underlying model for generation compatibility.

### JoraLayer

BaseTunerLayer implementation for JORA adaptation.

#### Constructor Parameters

- `base_layer: nn.Module` - Base layer to adapt (nn.Linear or Conv1D)
- `adapter_name: str` - Adapter name
- `cfg: JoraConfig` - JORA configuration

#### Methods

##### `forward(x: Tensor, *args, **kwargs) -> Tensor`
Forward pass with JORA adaptation.

**Parameters:**
- `x: Tensor` - Input tensor
- `*args, **kwargs` - Additional arguments passed to base layer

**Returns:**
- `Tensor` - Adapted output tensor

##### `compute_delta(x: Tensor) -> Tensor`
Compute the JORA adaptation delta.

**Parameters:**
- `x: Tensor` - Input tensor

**Returns:**
- `Tensor` - Adaptation delta

##### `maybe_apply_magnitude(out: Tensor) -> Tensor`
Apply magnitude scaling if configured.

**Parameters:**
- `out: Tensor` - Output tensor before magnitude scaling

**Returns:**
- `Tensor` - Scaled output tensor

##### `update_step(total_steps: int | None = None)`
Update parameter selection for this layer.

**Parameters:**
- `total_steps: int | None` - Total training steps for warmup calculation

##### `update_temperature(current_step: int, total_steps: int)`
Update temperature annealing for this layer.

**Parameters:**
- `current_step: int` - Current training step
- `total_steps: int` - Total training steps

##### `init_random_pairs(n_pairs_L: int | None = None, n_pairs_R: int | None = None)`
Initialize random parameter pairs for debugging/testing.

**Parameters:**
- `n_pairs_L: int | None` - Number of left-side pairs
- `n_pairs_R: int | None` - Number of right-side pairs

## Core Classes

### DiagCore

Diagonal core adaptation matrix.

#### Constructor Parameters

- `n: int` - Output dimension
- `m: int` - Input dimension
- `device` - Device for parameters
- `dtype` - Data type for parameters
- `zero_init: bool` - Initialize with zeros (default: False)

#### Methods

##### `forward() -> Tensor`
Generate diagonal adaptation matrix.

**Returns:**
- `Tensor` - Diagonal adaptation matrix of shape (n, m)

##### `get_row_slice(start: int, end: int) -> Tensor`
Get slice of diagonal matrix for sparse selection.

### BlockCore

Block-diagonal core adaptation matrix.

#### Constructor Parameters

- `n: int` - Output dimension
- `m: int` - Input dimension
- `device` - Device for parameters
- `dtype` - Data type for parameters
- `block_size: int` - Size of diagonal blocks (default: 4)
- `zero_init: bool` - Initialize with zeros (default: False)

#### Methods

##### `forward() -> Tensor`
Generate block-diagonal adaptation matrix.

##### `get_row_slice(start: int, end: int) -> Tensor`
Get slice of block-diagonal matrix.

### LowRankCore

Low-rank core adaptation using A@B^T factorization.

#### Constructor Parameters

- `n: int` - Output dimension
- `m: int` - Input dimension
- `device` - Device for parameters
- `dtype` - Data type for parameters
- `r: int` - Low-rank dimension (default: 8)
- `zero_init: bool` - Initialize with zeros (default: False)

#### Methods

##### `forward() -> Tensor`
Generate low-rank adaptation matrix A@B^T.

##### `get_row_slice(start: int, end: int) -> Tensor`
Get slice of low-rank matrix.

## Utility Functions

### rotation.py

##### `cayley_cos_sin(theta: Tensor) -> Tuple[Tensor, Tensor]`
Compute cos/sin using Cayley parameterization.

##### `apply_rotations(pairs: Tensor, thetas: Tensor, x: Tensor, rotation_param: str = "cayley") -> Tensor`
Apply sequence of 2D rotations to tensor.

### selection.py

##### `select_top_k_pairs_gpu(energy: Tensor, k: int, max_features: Optional[int] = None) -> Tensor`
Select top-k parameter pairs based on energy scores.

### magnitude.py

##### `compute_ecd_scale(base_row_norms: Tensor, total_energy: float, ecd_log_mag: Tensor, ecd_alpha: float, temperature: float, eps: float) -> Tensor`
Compute ECD (Energy-based Channel Dropping) scaling.

##### `compute_oer_scale_softmax(base_row_norms: Tensor, total_energy: float, oer_logits: Tensor, temperature: float, eps: float) -> Tensor`
Compute OER (Orthogonal Energy Redistribution) softmax scaling.

### utils.py

##### `get_in_out_features(module: nn.Module) -> Tuple[int, int]`
Get input and output feature dimensions for a module.

##### `linear_forward(module: nn.Module, x: torch.Tensor) -> torch.Tensor`
Unified forward pass for Linear and Conv1D layers.

## Usage Examples

### Basic Usage

```python
from peft import get_peft_model, JoraConfig

config = JoraConfig(
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    S_L=32, S_R=32, k=8
)
model = get_peft_model(model, config)
```

### Advanced Configuration

```python
config = JoraConfig(
    # Target modules
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],

    # Rotation settings
    S_L=64, S_R=64,
    rotation_param="cayley",

    # Selection settings
    k=16,
    selection="topk_ema",
    ema_beta=0.95,
    warmup_steps=100,

    # Core settings
    core="block",
    block_size=8,

    # Magnitude settings
    magnitude="oer_softmax",
    oer_temperature=2.0
)
```

### Ablation Studies

```python
# No rotation ablation
config_no_rotation = JoraConfig(S_L=0, S_R=0, k=8)

# No selection ablation
config_no_selection = JoraConfig(selection="none")

# Different core types
config_lowrank = JoraConfig(core="lowrank", lowrank_r=16)

# No magnitude scaling
config_no_magnitude = JoraConfig(magnitude="none")
```

## DDP Compatibility

JORA uses sparse per-step parameter selection. For DistributedDataParallel training, set:
```python
training_args = TrainingArguments(
    # ... other args
    ddp_find_unused_parameters=True
)
```

## Step Counter Semantics

JORA's internal step counter increments **per backward call (micro-batch)**, not per optimizer step.

- With gradient accumulation factor N, schedules progress N times faster
- For optimizer-step schedules, use larger `warmup_steps` or `warmup_ratio`

## Architecture Benefits

- **Modular Design**: Each component (rotation/selection/core/magnitude) is independently configurable
- **Research Friendly**: Easy to ablate individual components for ablation studies
- **Efficient**: Sparse selection reduces trainable parameters
- **Compatible**: Works with standard HF Trainer and PEFT ecosystem
