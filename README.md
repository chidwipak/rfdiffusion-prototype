# RFDiffusion Prototype for DeepChem (GSoC 2026)

This is my preparation project for Google Summer of Code 2026.
My goal is to bring advanced protein design models like RFDiffusion into DeepChem.

## What's in this repo?

I implemented a diffusion model that generates protein backbones. It works in 3 stages of complexity:
1.  **Basic Denoising**: Simple diffusion (like for images).
2.  **SE3-Equivariant**: Understands rotation and translation of molecules.
3.  **RoseTTAFold Backbone**: The full 3-track architecture used in the actual RFDiffusion paper.

## Quick Start

```bash
# 1. Setup
python -m venv venv
source venv/bin/activate
pip install torch numpy matplotlib pytest

# 2. Run All Unit Tests (38 tests)
python -m pytest tests/ -v

# 3. Run Memorization Sanity Check
python memorization_experiment.py --model se3 --proteins 5 --epochs 1000

# 4. Run the Tutorial (Generates a protein!)
python tutorials/rfdiffusion_tutorial.py

# 5. Comparisons (Runs analysis script)
python analysis.py
```

## ✅ Sanity Checks (Mentor's Request)

### Memorization Experiment
The mentor specifically asked: *"Can you memorize a tiny dataset? That is usually a good sanity check (5 to 10 datapoints)."*

I implemented `memorization_experiment.py` which:
- Creates a tiny dataset of 5 proteins with realistic geometry
- Trains the model until it can reproduce the training data
- Verifies physical validity (Ca-Ca distances ≈ 3.8 Å)

```bash
# Run the memorization sanity check
python memorization_experiment.py --model se3 --proteins 5 --epochs 1000

# Compare Basic MLP vs SE3 Model
python memorization_experiment.py --model compare
```

### Unit Tests (38 tests, all passing ✓)
```bash
python -m pytest tests/ -v
```

Tests cover:
- `test_backbone_diffusion.py` - Diffusion schedule, noise prediction
- `test_rosettafold.py` - 3-track architecture shapes
- `test_se3_diffusion.py` - SE3 equivariance, frame representation
- `test_conditioning.py` - Motif, length, and binder conditioning
- `test_memorization.py` - Tiny dataset creation, geometry validation

## Code Example

```python
from se3_diffusion import SE3DiffusionDenoiser
from backbone_diffusion import DiffusionSchedule

# Initialize the 3-track RoseTTAFold model
model = SE3DiffusionDenoiser(
    embed_dim=128,
    num_layers=4,
    num_heads=4
)

# Create diffusion schedule
schedule = DiffusionSchedule(num_timesteps=100)

# Forward pass: predict noise
coords = torch.randn(4, 50, 9)  # (batch, residues, 3 atoms × 3 coords)
t = torch.randint(0, 100, (4,))  # timesteps
noise_pred = model(coords, t)
```

## Project Structure

```
rfdiffusion-prototype/
├── se3_diffusion.py         # Main SE3 diffusion denoiser
├── rosettafold.py           # RoseTTAFold 3-track architecture
├── se3_layers.py            # Invariant Point Attention
├── frame_representation.py  # SE(3) frame utilities
├── backbone_diffusion.py    # Basic DDPM implementation
├── conditioning.py          # Motif, length, binder conditioning
├── cath_loader.py           # CATH dataset loader
├── train_cath.py            # Training script for CATH
├── memorization_experiment.py  # Tiny dataset sanity check
├── analysis.py              # Model comparison analysis
├── tests/                   # Unit tests (38 tests)
│   ├── test_backbone_diffusion.py
│   ├── test_rosettafold.py
│   ├── test_se3_diffusion.py
│   ├── test_conditioning.py
│   └── test_memorization.py
└── tutorials/
    └── rfdiffusion_tutorial.py
```

## Key Components

| Module | Description | Lines |
|--------|-------------|-------|
| `se3_diffusion.py` | SE3-equivariant denoiser with RoseTTAFold backbone | ~320 |
| `rosettafold.py` | 3-track architecture (1D, 2D, 3D) | ~200 |
| `se3_layers.py` | Invariant Point Attention implementation | ~210 |
| `conditioning.py` | Motif, length, binder conditioning | ~350 |
| `backbone_diffusion.py` | DDPM schedule and basic denoiser | ~380 |

## Training on CATH Dataset

I trained for 500 epochs on ~30 CATH domains:
- Loss decreased from ~1.2 to ~0.88 (27% reduction)
- Training time: ~26 minutes on CPU

```bash
# Run CATH training
python train_cath.py --epochs 100
```

## Status

- [x] Basic Prototype (DDPM)
- [x] SE3 Layers (Invariant Point Attention)
- [x] RoseTTAFold 3-Track Architecture
- [x] Conditioning (Motifs, Length, Binder)
- [x] Unit Tests (38 tests passing)
- [x] Memorization Experiment (Sanity Check)
- [x] CATH Dataset Training
- [x] NumPy-style Docstrings
- [x] Type Annotations
- [ ] Full PDB-scale training (needs more compute)
