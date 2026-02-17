# RFDiffusion Prototype (GSoC 2026 Prep)

Prototype work for implementing RFDiffusion in DeepChem. This repo contains my initial experiments and architecture exploration. The proper DeepChem integration is in my [deepchem fork](https://github.com/chidwipak/deepchem/tree/gsoc-cath-integration).

## what this is

a diffusion model for protein backbone generation, built as prep work for gsoc 2026 with deepchem. i implemented this at 3 levels of complexity:

1. **basic ddpm** - simple denoising diffusion for protein coordinates
2. **se3 equivariant** - handles rotations and translations properly
3. **rosettafold backbone** - 3-track architecture from the actual paper

## setup

```bash
python -m venv venv
source venv/bin/activate
pip install torch numpy matplotlib pytest
```

## quick start

```bash
# run all tests
python -m pytest tests/ -v

# memorization sanity check
python memorization_experiment.py --model se3 --proteins 5 --epochs 1000

# train on cath
python train_cath.py --epochs 100

# tutorial
python tutorials/rfdiffusion_tutorial.py
```

## training results

trained on cath domains with multi-gpu (3x k80):

| metric | value |
|--------|-------|
| best loss | 0.033 |
| loss reduction | 96%+ |
| gpus | 3x k80 (ddp) |

memorization test (mentor's sanity check):
- 5 proteins, 500 epochs
- final loss: 0.018
- rmsd: 0.025 angstrom

## deepchem integration

the actual deepchem integration is in my fork — the model is now a proper `TorchModel` subclass:

```python
import deepchem as dc

tasks, datasets, _ = dc.molnet.load_cath(featurizer='ProteinBackbone')
train, valid, test = datasets

model = dc.models.RFDiffusionModel(embed_dim=256, num_layers=8)
model.fit(train, nb_epoch=100)

samples = model.generate(num_samples=5, seq_length=50)
```

see the [deepchem fork](https://github.com/chidwipak/deepchem/tree/gsoc-cath-integration) for the integrated code.

## files

```
├── se3_diffusion.py          # se3 equivariant denoiser
├── rosettafold.py            # 3-track architecture
├── se3_layers.py             # invariant point attention
├── backbone_diffusion.py     # basic ddpm
├── conditioning.py           # motif, length, binder conditioning
├── frame_representation.py   # se3 frame utilities
├── cath_loader.py            # cath dataset loader
├── train_cath.py             # cath training script
├── train_improved.py         # multi-gpu training
├── memorization_experiment.py # tiny dataset sanity check
├── tests/                    # unit tests
└── tutorials/                # usage examples
```

## mentor checklist

- [x] memorization test (5-10 samples)
- [x] numpy style docstrings
- [x] type annotations
- [x] usage examples
- [x] unit tests
- [x] deepchem dataset integration (TorchModel subclass)
- [ ] scale to full pdb (needs more compute, gsoc goal)
