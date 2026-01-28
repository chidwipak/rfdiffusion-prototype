# RFDiffusion Prototype for DeepChem

A working implementation of RFDiffusion-style protein structure generation, built as preparation for GSoC 2026.

## What This Is

This prototype demonstrates core concepts from RFDiffusion paper:
- **DDPM diffusion** for protein backbone coordinates
- **SE3-equivariant denoiser** with Invariant Point Attention
- **Conditioning mechanisms** (motif, length, binder)
- **TorchModel integration** following DeepChem patterns

## Results

Trained two models for comparison (50 epochs each):

| Model      | Parameters | Loss (start → end) | Rotation Consistency |
| ---------- | ---------- | ------------------ | -------------------- |
| Basic DDPM | 1.2M       | 1.38 → 0.29        | 0.132 error          |
| SE3 Model  | 950K       | 1.41 → 0.64        | 0.097 error          |

SE3 model uses fewer parameters and is 1.4x more rotation-consistent.

## Files

| File                          | Description                          |
| ----------------------------- | ------------------------------------ |
| `backbone_diffusion.py`       | Core DDPM with sinusoidal embeddings |
| `backbone_diffusion_model.py` | DeepChem TorchModel wrapper          |
| `protein_coords_loader.py`    | PDB parsing utilities                |
| `frame_representation.py`     | N-Ca-C → rotation + translation      |
| `se3_diffusion.py`            | IPA-based SE3 denoiser               |
| `conditioning.py`             | Motif, length, binder conditioning   |
| `train_prototype.py`          | Basic training script                |
| `train_se3.py`                | SE3 model training                   |
| `analysis.py`                 | Comparison script                    |

## Quick Start

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install torch numpy

# Run analysis (compares both models)
python analysis.py

# Train basic model
python train_prototype.py --epochs 50

# Train SE3 model  
python train_se3.py --epochs 100

# Quick test
python train_se3.py --test-mode
```

## Technical Details

### Frame Representation
Converts N-Ca-C backbone atoms to rotation matrix + translation:
```
N, Ca, C coordinates → (R ∈ SO(3), t ∈ ℝ³)
```

### Invariant Point Attention
Combines standard attention with distance-based geometric attention for rotation consistency.

### Conditioning
All three types work as additive embeddings and can be combined.

## Author

Chidwipak 

## References

- RFDiffusion: "De novo design of protein structure and function" (Nature 2023)
- AlphaFold2: IPA architecture
- DDPM: "Denoising Diffusion Probabilistic Models" (Ho et al. 2020)
