"""
Training Script for SE3 Diffusion Prototype

This script trains the SE3-aware diffusion model for protein backbone
coordinate generation and compares with the basic MLP model.

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

from backbone_diffusion import DiffusionSchedule
from se3_diffusion import SE3DiffusionDenoiser, compute_se3_diffusion_loss


def generate_synthetic_data(num_proteins: int = 50, num_residues: int = 50):
    """Generate synthetic protein-like backbone coordinates."""
    coords_list = []
    
    for _ in range(num_proteins):
        coords = np.zeros((num_residues, 9), dtype=np.float32)
        current_pos = np.array([0.0, 0.0, 0.0])
        
        for i in range(num_residues):
            theta = np.random.uniform(0, 2 * np.pi)
            phi = np.random.uniform(0, np.pi)
            
            n_pos = current_pos.copy()
            direction = np.array([
                np.sin(phi) * np.cos(theta),
                np.sin(phi) * np.sin(theta),
                np.cos(phi)
            ])
            direction = direction / np.linalg.norm(direction + 1e-8)
            ca_pos = n_pos + direction * 1.46
            
            theta2 = theta + np.random.uniform(-0.5, 0.5)
            phi2 = phi + np.random.uniform(-0.5, 0.5)
            direction2 = np.array([
                np.sin(phi2) * np.cos(theta2),
                np.sin(phi2) * np.sin(theta2),
                np.cos(phi2)
            ])
            direction2 = direction2 / np.linalg.norm(direction2 + 1e-8)
            c_pos = ca_pos + direction2 * 1.52
            
            coords[i, 0:3] = n_pos
            coords[i, 3:6] = ca_pos
            coords[i, 6:9] = c_pos
            
            current_pos = c_pos + direction2 * 1.33
        
        # Normalize
        coords = coords - coords.mean(axis=0)
        scale = np.abs(coords).max() + 1e-8
        coords = coords / scale
        
        coords_list.append(coords)
    
    return np.array(coords_list)


def train(args):
    """Main training function for SE3 diffusion."""
    print("=" * 60)
    print("SE3 Diffusion Prototype Training")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Generate synthetic data
    print("\nGenerating synthetic data...")
    X = generate_synthetic_data(args.num_proteins, args.num_residues)
    print(f"Training data shape: {X.shape}")
    
    # Convert to tensor
    X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
    
    # Create model
    print("\nCreating SE3 Diffusion model...")
    model = SE3DiffusionDenoiser(
        embed_dim=args.embed_dim,
        time_dim=args.time_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads
    ).to(device)
    
    # Create schedule
    schedule = DiffusionSchedule(
        num_timesteps=args.num_timesteps,
        device=device
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Train
    print(f"\nTraining for {args.epochs} epochs...")
    print("-" * 40)
    
    losses = []
    model.train()
    
    for epoch in range(args.epochs):
        optimizer.zero_grad()
        
        # Sample batch
        if args.batch_size < X_tensor.shape[0]:
            indices = torch.randperm(X_tensor.shape[0])[:args.batch_size]
            batch = X_tensor[indices]
        else:
            batch = X_tensor
        
        # Compute loss
        loss = compute_se3_diffusion_loss(model, batch, schedule)
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        losses.append(loss.item())
        
        if (epoch + 1) % max(1, args.epochs // 10) == 0 or epoch == 0:
            print(f"Epoch {epoch + 1:4d}/{args.epochs} | Loss: {loss.item():.6f}")
    
    print("-" * 40)
    print(f"Final loss: {losses[-1]:.6f}")
    
    # Check learning
    if len(losses) > 10:
        initial_avg = np.mean(losses[:5])
        final_avg = np.mean(losses[-5:])
        improvement = (initial_avg - final_avg) / initial_avg * 100
        print(f"Loss improvement: {improvement:.1f}%")
        
        if improvement > 10:
            print("✓ Model is learning!")
        elif improvement > 0:
            print("~ Model is learning slowly, may need more epochs")
        else:
            print("✗ Loss did not decrease. May need tuning.")
    
    # Test generation
    print("\nTesting sample generation...")
    model.eval()
    with torch.no_grad():
        # Sample from noise
        shape = (2, args.num_residues, 9)
        x = torch.randn(shape, device=device)
        
        # Denoise with reduced steps for speed
        num_steps = min(50, args.num_timesteps)
        for t in reversed(range(num_steps)):
            t_batch = torch.full((2,), t, device=device, dtype=torch.long)
            noise_pred = model(x, t_batch)
            
            # Simple Euler step (approximate)
            sigma = schedule.sqrt_one_minus_alpha_cumprod[t]
            x = x - 0.1 * noise_pred * sigma
        
        print(f"Generated samples shape: {x.shape}")
        print(f"Sample coordinate range: [{x.min():.3f}, {x.max():.3f}]")
    
    # Save checkpoint
    if args.save_path:
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'config': {
                'embed_dim': args.embed_dim,
                'time_dim': args.time_dim,
                'num_layers': args.num_layers,
                'num_heads': args.num_heads,
                'num_timesteps': args.num_timesteps
            },
            'final_loss': losses[-1]
        }
        torch.save(checkpoint, args.save_path)
        print(f"\nCheckpoint saved to {args.save_path}")
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    
    return losses


def main():
    parser = argparse.ArgumentParser(
        description="Train SE3 diffusion model for protein structure generation"
    )
    
    # Data arguments
    parser.add_argument('--num-proteins', type=int, default=50)
    parser.add_argument('--num-residues', type=int, default=30)
    
    # Model arguments  
    parser.add_argument('--embed-dim', type=int, default=128)
    parser.add_argument('--time-dim', type=int, default=64)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--num-heads', type=int, default=4)
    parser.add_argument('--num-timesteps', type=int, default=100)
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--save-path', type=str, default=None)
    
    # Quick test mode
    parser.add_argument('--test-mode', action='store_true')
    
    args = parser.parse_args()
    
    if args.test_mode:
        args.num_proteins = 10
        args.num_residues = 20
        args.num_timesteps = 50
        args.embed_dim = 64
        args.num_layers = 1
        args.epochs = 30
        print("Running in test mode...")
    
    train(args)


if __name__ == "__main__":
    main()
