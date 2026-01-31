# RFDiffusion Prototype - GSoC 2026# RFDiffusion Prototype for DeepChem (GSoC 2026)



hey! this is my prep work for gsoc 2026 with deepchem. building a diffusion model for protein backbone generation (like rfdiffusion).This is my preparation project for Google Summer of Code 2026.

My goal is to bring advanced protein design models like RFDiffusion into DeepChem.

## quick start

## What's in this repo?

```bash

# setupI implemented a diffusion model that generates protein backbones. It works in 3 stages of complexity:

source venv/bin/activate1.  **Basic Denoising**: Simple diffusion (like for images).

pip install torch numpy pytest2.  **SE3-Equivariant**: Understands rotation and translation of molecules.

3.  **RoseTTAFold Backbone**: The full 3-track architecture used in the actual RFDiffusion paper.

# run tests (38 tests)

python -m pytest tests/ -v## Quick Start



# run memorization sanity check```bash

python run_memorization_test.py# 1. Setup

python -m venv venv

# train on cathsource venv/bin/activate

python train_improved.pypip install torch numpy matplotlib pytest

```

# 2. Run All Unit Tests (38 tests)

## what i builtpython -m pytest tests/ -v



3 levels of complexity:# 3. Run Memorization Sanity Check

1. basic ddpm - simple denoising python memorization_experiment.py --model se3 --proteins 5 --epochs 1000

2. se3 equivariant - rotation/translation invariant

3. rosettafold backbone - full 3-track architecture# 4. Run the Tutorial (Generates a protein!)

python tutorials/rfdiffusion_tutorial.py

## training results

# 5. Comparisons (Runs analysis script)

trained on cath domains with multi-gpu (3x tesla k80):python analysis.py

```

| metric | value |

|--------|-------|## ✅ Sanity Checks (Mentor's Request)

| best loss | 0.033 |

| epochs | 1000 |### Memorization Experiment

| loss reduction | 96.6% |The mentor specifically asked: *"Can you memorize a tiny dataset? That is usually a good sanity check (5 to 10 datapoints)."*

| gpus | 3x k80 (ddp) |

I implemented `memorization_experiment.py` which:

memorization test (mentor's sanity check):- Creates a tiny dataset of 5 proteins with realistic geometry

- 5 proteins, 500 epochs- Trains the model until it can reproduce the training data

- final loss: 0.018- Verifies physical validity (Ca-Ca distances ≈ 3.8 Å)

- rmsd: 0.025

- passed ✓```bash

# Run the memorization sanity check

## filespython memorization_experiment.py --model se3 --proteins 5 --epochs 1000



```# Compare Basic MLP vs SE3 Model

├── improved_diffusion.py    # transformer denoiser (per-atom noise)python memorization_experiment.py --model compare

├── train_improved.py        # multi-gpu training script  ```

├── run_memorization_test.py # sanity check script

├── se3_diffusion.py         # se3 equivariant model### Unit Tests (38 tests, all passing ✓)

├── rosettafold.py           # 3-track architecture```bash

├── se3_layers.py            # invariant point attentionpython -m pytest tests/ -v

├── conditioning.py          # motif/length/binder conditioning```

├── tests/                   # 38 unit tests

└── checkpoints_v2/          # trained weightsTests cover:

```- `test_backbone_diffusion.py` - Diffusion schedule, noise prediction

- `test_rosettafold.py` - 3-track architecture shapes

## tests- `test_se3_diffusion.py` - SE3 equivariance, frame representation

- `test_conditioning.py` - Motif, length, and binder conditioning

38 tests covering:- `test_memorization.py` - Tiny dataset creation, geometry validation

- diffusion schedule (forward/reverse)

- noise prediction## Code Example

- se3 equivariance

- conditioning modules```python

- memorizationfrom se3_diffusion import SE3DiffusionDenoiser

from backbone_diffusion import DiffusionSchedule

```bash

python -m pytest tests/ -v# Initialize the 3-track RoseTTAFold model

# all 38 passmodel = SE3DiffusionDenoiser(

```    embed_dim=128,

    num_layers=4,

## mentor checklist    num_heads=4

)

from bharath's feedback:

# Create diffusion schedule

- [x] memorization test (5-10 samples) - done, passesschedule = DiffusionSchedule(num_timesteps=100)

- [x] numpy doc style - all modules documented

- [x] type annotations - all functions typed  # Forward pass: predict noise

- [x] usage examples - tutorials foldercoords = torch.randn(4, 50, 9)  # (batch, residues, 3 atoms × 3 coords)

- [x] unit tests - 38 tests passingt = torch.randint(0, 100, (4,))  # timesteps

noise_pred = model(coords, t)

## next steps```



- [ ] pdb dataset integration (need more compute)## Project Structure

- [ ] scale up training

- [ ] add more conditioning types```

rfdiffusion-prototype/

---├── se3_diffusion.py         # Main SE3 diffusion denoiser

gsoc 2026 prep - deepchem rfdiffusion project├── rosettafold.py           # RoseTTAFold 3-track architecture

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
