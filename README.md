# RFDiffusion Prototype for DeepChem (GSoC 2026)

This is my preparation project for Google Summer of Code 2026.
My goal is to bring advanced protein design models like RFDiffusion into DeepChem.

## What's in this repo?

I implemented a diffusion model that generates protein backbones. It works in 3 stages of complexity:
1.  **Basic Denoising**: Simple diffusion (like for images).
2.  **SE3-Equivariant**: Understands rotation and translation of molecules.
3.  **RoseTTAFold Backbone**: The full 3-track architecture used in the actual RFDiffusion paper.

## How to Run

I made sure this is easy to run. Just set up a virtual environment and use the tutorial.

```bash
# 1. Setup
python -m venv venv
source venv/bin/activate
pip install torch numpy matplotlib

# 2. Run the Tutorial (Generates a protein!)
python tutorials/rfdiffusion_tutorial.py

# 3. Comparisons (Runs analysis script)
python analysis.py
```

## Code Example

Want to use the model in your own script?

```python
from rfdiffusion_prototype.se3_diffusion import SE3DiffusionDenoiser

# Initialize the 3-track RoseTTAFold model
model = SE3DiffusionDenoiser(
    embed_dim=128,
    num_layers=4
).cuda()

# Pass in noisy coordinates (Batch, Residues, 9)
noise_pred = model(coords, timesteps)
```

## Structure

- `rfdiffusion-prototype/`: The core package.
    - `rosettafold.py`: The RoseTTAFold architecture.
    - `se3_diffusion.py`: The main diffusion logic.
- `tutorials/`: Example scripts showing how it works.

## Status

- [x] Basic Prototype
- [x] SE3 Layers (Invariant Point Attention)
- [x] Conditioning (Motifs, Length)
- [x] Full RoseTTAFold Backbone
- [ ] Benchmarking on CATH dataset (Doing this next!)
