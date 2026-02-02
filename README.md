# protein diffusion model - gsoc 2026 prep
# RFDiffusion Prototype for DeepChem (GSoC 2026)



building this for my deepchem gsoc application



basically trying to replicate rfdiffusion but on smaller scale since i dont have their compute resourceshey! this is my prep work for gsoc 2026 with deepchem. building a diffusion model for protein backbone generation (like rfdiffusion).This is my preparation project for Google Summer of Code 2026.



## setupMy goal is to bring advanced protein design models like RFDiffusion into DeepChem.



```bash## quick start

pip install torch numpy pytest

## What's in this repo?

# run tests

pytest tests/ -v```bash



# memorization test# setupI implemented a diffusion model that generates protein backbones. It works in 3 stages of complexity:

python run_memorization_test.py

source venv/bin/activate1.  **Basic Denoising**: Simple diffusion (like for images).

# full training

python train_improved.pypip install torch numpy pytest2.  **SE3-Equivariant**: Understands rotation and translation of molecules.

```

3.  **RoseTTAFold Backbone**: The full 3-track architecture used in the actual RFDiffusion paper.

## what ive built

# run tests (38 tests)

started with basic stuff then kept adding complexity

python -m pytest tests/ -v## Quick Start

1. simple diffusion (ddpm style but for protein coordinates)

2. se3 equivariant layers - handles rotations/translations properly  

3. rosettafold inspired architecture - 3 track system

# run memorization sanity check```bash

## recent training results

python run_memorization_test.py# 1. Setup

just ran this yesterday, took about 12hrs on our 3 gpus

python -m venv venv

dataset: ~30 cath domain proteins

- initial loss: 0.97# train on cathsource venv/bin/activate

- best loss: 0.033 (at epoch 770)

- final: 0.083python train_improved.pypip install torch numpy matplotlib pytest

- reduction: 96%

```

also tested on tiny dataset (5 proteins) to verify it actually learns:

- loss: 1.0 -> 0.018# 2. Run All Unit Tests (38 tests)

- rmsd: 0.025 angstrom

- basically it can memorize which means architecture works## what i builtpython -m pytest tests/ -v



## comparison with actual rfdiffusion



their setup:3 levels of complexity:# 3. Run Memorization Sanity Check

- 150k+ proteins from pdb

- fine tuned from pretrained rosettafold  1. basic ddpm - simple denoising python memorization_experiment.py --model se3 --proteins 5 --epochs 1000

- fancy gpus (probably a100s)

- weeks of training2. se3 equivariant - rotation/translation invariant



my setup:3. rosettafold backbone - full 3-track architecture# 4. Run the Tutorial (Generates a protein!)

- 30 proteins from cath

- training from scratchpython tutorials/rfdiffusion_tutorial.py

- 3x tesla k80s

- ~12 hours## training results



same core ideas tho - mse loss, diffusion process, se3 equivariance# 5. Comparisons (Runs analysis script)



hard to compare losses directly since paper doesnt report training curves. they focus on downstream stuff like af2 validationtrained on cath domains with multi-gpu (3x tesla k80):python analysis.py



## files```



```| metric | value |

improved_diffusion.py - main model (transformer based, 7.5M params)

train_improved.py - training with pytorch ddp|--------|-------|## ✅ Sanity Checks (Mentor's Request)

run_memorization_test.py - sanity check test

se3_diffusion.py - se3 equivariant version| best loss | 0.033 |

rosettafold.py - 3 track architecture

se3_layers.py - invariant point attention| epochs | 1000 |### Memorization Experiment

tests/ - 38 unit tests

checkpoints_v2/ - saved models & metrics| loss reduction | 96.6% |The mentor specifically asked: *"Can you memorize a tiny dataset? That is usually a good sanity check (5 to 10 datapoints)."*

```

| gpus | 3x k80 (ddp) |

## mentor feedback

I implemented `memorization_experiment.py` which:

bharath asked me to do:

- [x] memorization test on 5-10 proteinsmemorization test (mentor's sanity check):- Creates a tiny dataset of 5 proteins with realistic geometry

- [x] numpy style docstrings  

- [x] type annotations- 5 proteins, 500 epochs- Trains the model until it can reproduce the training data

- [x] unit tests

- final loss: 0.018- Verifies physical validity (Ca-Ca distances ≈ 3.8 Å)

all done, 38 tests passing

- rmsd: 0.025

## whats next

- passed ✓```bash

if i get into gsoc:

- scale to full pdb dataset# Run the memorization sanity check

- add self conditioning (like in paper)

- integrate with deepchem loaders## filespython memorization_experiment.py --model se3 --proteins 5 --epochs 1000

- more conditioning types (secondary structure etc)



this is proof of concept showing i understand the mechanisms

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
