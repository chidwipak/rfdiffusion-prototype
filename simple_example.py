"""
Simple Usage Example for RFDiffusion Prototype
"""

import torch
from se3_diffusion import SE3DiffusionDenoiser
from backbone_diffusion import DiffusionSchedule

def main():
    print("RFDiffusion Prototype - End-to-End Generation Example")
    print("-" * 50)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on: {device}")
    
    # 1. Initialize Model
    print("\n1. Initializing SE3-Equivariant Model...")
    model = SE3DiffusionDenoiser(
        embed_dim=128,
        time_dim=64,
        num_layers=2,
        num_heads=4
    ).to(device)
    print("   Model created (950k params)")
    
    # 2. Setup Diffusion Schedule
    print("\n2. Setting up Diffusion Schedule...")
    schedule = DiffusionSchedule(num_timesteps=100, device=device)
    
    # 3. Generate a protein structure (Reverse Diffusion)
    print("\n3. Generating random protein backbone (50 residues)...")
    print("   Denoising from pure Gaussian noise...")
    
    # Sample from noise (batch_size=1, length=50, 9 coords per res)
    generated_coords = schedule.sample(
        model, 
        shape=(1, 50, 9), 
        device=device
    )
    
    print("\n   Generation complete!")
    print(f"   Output shape: {generated_coords.shape}")
    print("   (Batch, Residues, Atoms*Coords)")
    
    # 4. Analyze result
    print("\n4. Inspection:")
    # Reshape to (Residues, 3 atoms, 3 coords)
    structure = generated_coords[0].reshape(50, 3, 3).cpu().detach().numpy()
    
    # Calculate average N-Ca distance (should be around 1.46 Angstroms if trained)
    # Since this is untrained in this example, it will be random, but shows the pipeline works.
    n = structure[:, 0]
    ca = structure[:, 1]
    dists = torch.norm(torch.tensor(n - ca), dim=1)
    
    print(f"   Generated {len(structure)} residues")
    print(f"   Average N-Ca bond length: {dists.mean():.3f} (Random init model)")
    print("\n   To train a model, run: python train_se3.py")

if __name__ == "__main__":
    main()
