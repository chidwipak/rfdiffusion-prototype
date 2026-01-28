"""
Training Script for Backbone Diffusion Prototype

This script demonstrates training a diffusion model for protein backbone
coordinates using DeepChem's TorchModel framework.

Author: Chidwipak
Date: January 2026
"""

import os
import sys
import argparse
import numpy as np
import torch
from pathlib import Path

# Add prototype directory to path
sys.path.insert(0, str(Path(__file__).parent))

from backbone_diffusion import BackboneDiffusionDenoiser, DiffusionSchedule, compute_diffusion_loss
from backbone_diffusion_model import BackboneDiffusionModel
from protein_coords_loader import (
    parse_pdb_backbone,
    load_pdb_as_flat_coords,
    create_diffusion_dataset,
    find_pdb_files,
    pad_or_truncate
)


def get_test_pdb_files():
    """Get PDB files from DeepChem's test data."""
    # Look in typical DeepChem installation paths
    possible_paths = [
        "/home/chidwipak/Gsoc2026/deepchem/deepchem/feat/tests/data",
        os.path.expanduser("~/deepchem/deepchem/feat/tests/data"),
        "./deepchem/deepchem/feat/tests/data",
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            pdb_files = find_pdb_files(path)
            if pdb_files:
                return pdb_files
    
    return []


def generate_synthetic_data(num_proteins: int = 50, num_residues: int = 50):
    """
    Generate synthetic protein-like backbone coordinates for testing.
    Creates coordinates that roughly follow natural backbone geometry.
    """
    coords_list = []
    
    for _ in range(num_proteins):
        # Start at origin
        coords = np.zeros((num_residues, 9), dtype=np.float32)
        
        # Generate backbone with realistic inter-atom distances
        # N-Ca: ~1.46 Å, Ca-C: ~1.52 Å, C-N: ~1.33 Å
        current_pos = np.array([0.0, 0.0, 0.0])
        
        for i in range(num_residues):
            # Random direction for chain growth
            theta = np.random.uniform(0, 2 * np.pi)
            phi = np.random.uniform(0, np.pi)
            
            # N atom
            n_pos = current_pos.copy()
            
            # Ca atom (1.46 Å from N)
            direction = np.array([
                np.sin(phi) * np.cos(theta),
                np.sin(phi) * np.sin(theta),
                np.cos(phi)
            ]) + np.random.randn(3) * 0.1
            direction = direction / np.linalg.norm(direction)
            ca_pos = n_pos + direction * 1.46
            
            # C atom (1.52 Å from Ca)
            theta2 = theta + np.random.uniform(-0.5, 0.5)
            phi2 = phi + np.random.uniform(-0.5, 0.5)
            direction2 = np.array([
                np.sin(phi2) * np.cos(theta2),
                np.sin(phi2) * np.sin(theta2),
                np.cos(phi2)
            ]) + np.random.randn(3) * 0.1
            direction2 = direction2 / np.linalg.norm(direction2)
            c_pos = ca_pos + direction2 * 1.52
            
            # Store flattened coordinates
            coords[i, 0:3] = n_pos
            coords[i, 3:6] = ca_pos
            coords[i, 6:9] = c_pos
            
            # Move to next residue (C-N bond ~1.33 Å)
            current_pos = c_pos + direction2 * 1.33
        
        # Normalize
        coords = coords - coords.mean(axis=0)
        scale = np.abs(coords).max()
        if scale > 0:
            coords = coords / scale
        
        coords_list.append(coords)
    
    return np.array(coords_list)


def train(args):
    """Main training function."""
    print("=" * 60)
    print("Backbone Diffusion Prototype Training")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Load data
    print("\nLoading data...")
    pdb_files = get_test_pdb_files()
    
    if pdb_files and not args.synthetic:
        print(f"Found {len(pdb_files)} PDB files")
        try:
            dataset = create_diffusion_dataset(
                pdb_files,
                target_length=args.num_residues,
                normalize=True
            )
            if isinstance(dataset, dict):
                X = dataset['X']
            else:
                X = dataset.X
            print(f"Dataset shape: {X.shape}")
        except Exception as e:
            print(f"Error loading PDB files: {e}")
            print("Falling back to synthetic data...")
            X = generate_synthetic_data(args.num_proteins, args.num_residues)
    else:
        if not args.synthetic:
            print("No PDB files found. Using synthetic data.")
        else:
            print("Using synthetic data as requested.")
        X = generate_synthetic_data(args.num_proteins, args.num_residues)
    
    print(f"Training data shape: {X.shape}")
    print(f"  - {X.shape[0]} proteins")
    print(f"  - {X.shape[1]} residues")
    print(f"  - {X.shape[2]} coordinates (N, Ca, C flattened)")
    
    # Create dataset
    try:
        import deepchem as dc
        y = np.zeros((X.shape[0], 1), dtype=np.float32)
        dataset = dc.data.NumpyDataset(X=X, y=y)
        use_deepchem = True
    except ImportError:
        print("DeepChem not available. Using standalone mode.")
        dataset = {'X': X, 'y': np.zeros((X.shape[0], 1))}
        use_deepchem = False
    
    # Create model
    print("\nCreating model...")
    model = BackboneDiffusionModel(
        num_timesteps=args.num_timesteps,
        hidden_dim=args.hidden_dim,
        time_dim=args.time_dim,
        num_layers=args.num_layers,
        learning_rate=args.learning_rate,
        device=torch.device(device)
    )
    
    # Count parameters
    num_params = sum(p.numel() for p in model.model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Train
    print(f"\nTraining for {args.epochs} epochs...")
    print("-" * 40)
    
    losses = []
    for epoch in range(args.epochs):
        model.model.train()
        
        # Manual training loop for standalone mode
        if use_deepchem:
            epoch_losses = []
            for batch in dataset.iterbatches(batch_size=args.batch_size):
                coords, _, _ = model._prepare_batch(batch)
                model.optimizer.zero_grad()
                loss = compute_diffusion_loss(model.model, coords, model.schedule)
                loss.backward()
                model.optimizer.step()
                epoch_losses.append(loss.item())
            avg_loss = np.mean(epoch_losses)
        else:
            X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
            model.optimizer.zero_grad()
            loss = compute_diffusion_loss(model.model, X_tensor, model.schedule)
            loss.backward()
            model.optimizer.step()
            avg_loss = loss.item()
        
        losses.append(avg_loss)
        
        if (epoch + 1) % max(1, args.epochs // 10) == 0 or epoch == 0:
            print(f"Epoch {epoch + 1:4d}/{args.epochs} | Loss: {avg_loss:.6f}")
    
    print("-" * 40)
    print(f"Final loss: {losses[-1]:.6f}")
    
    # Check if loss decreased
    if len(losses) > 1:
        improvement = (losses[0] - losses[-1]) / losses[0] * 100
        print(f"Loss improvement: {improvement:.1f}%")
        
        if losses[-1] < losses[0]:
            print("✓ Model is learning!")
        else:
            print("✗ Loss did not decrease. May need more epochs or tuning.")
    
    # Generate samples
    print("\nGenerating samples...")
    model.model.eval()
    with torch.no_grad():
        samples = model.generate(num_samples=2, num_residues=args.num_residues)
    print(f"Generated samples shape: {samples.shape}")
    print(f"Sample coordinate range: [{samples.min():.3f}, {samples.max():.3f}]")
    
    # Save checkpoint
    if args.save_path:
        model.save_checkpoint(args.save_path)
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    
    return losses


def main():
    parser = argparse.ArgumentParser(
        description="Train backbone diffusion model for protein structure generation"
    )
    
    # Data arguments
    parser.add_argument('--num-proteins', type=int, default=50,
                        help='Number of synthetic proteins to generate')
    parser.add_argument('--num-residues', type=int, default=50,
                        help='Number of residues per protein')
    parser.add_argument('--synthetic', action='store_true',
                        help='Use synthetic data instead of PDB files')
    
    # Model arguments
    parser.add_argument('--num-timesteps', type=int, default=100,
                        help='Number of diffusion timesteps')
    parser.add_argument('--hidden-dim', type=int, default=128,
                        help='Hidden dimension of denoiser')
    parser.add_argument('--time-dim', type=int, default=64,
                        help='Dimension of time embeddings')
    parser.add_argument('--num-layers', type=int, default=3,
                        help='Number of residual blocks')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--save-path', type=str, default=None,
                        help='Path to save checkpoint')
    
    # Quick test mode
    parser.add_argument('--test-mode', action='store_true',
                        help='Run quick test with minimal settings')
    
    args = parser.parse_args()
    
    # Override for test mode
    if args.test_mode:
        args.num_proteins = 10
        args.num_residues = 20
        args.num_timesteps = 50
        args.hidden_dim = 64
        args.num_layers = 2
        args.epochs = 20
        args.synthetic = True
        print("Running in test mode with minimal settings...")
    
    train(args)


if __name__ == "__main__":
    main()
