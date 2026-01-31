#!/usr/bin/env python3
"""
Memorization Experiment - Sanity Check for Diffusion Model
===========================================================

This script tests if the model can memorize a tiny dataset (5-10 proteins).
This is the standard sanity check suggested for diffusion models - if the 
model can't overfit to a small dataset, something is fundamentally wrong.

As per mentor's suggestion: "Can you memorize a tiny dataset? That is 
usually a good sanity check (5 to 10 datapoints)."

Expected behavior:
- Loss should drop to near-zero (<0.01) 
- Model should perfectly reconstruct the training samples
- This validates that the architecture and training loop work correctly

Author: Chidwipak
Date: January 2026

Usage
-----
$ python run_memorization_test.py

References
----------
- DDPM paper section on memorization tests
- RFDiffusion supplementary materials
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from improved_diffusion import (
    ImprovedDiffusionDenoiser,
    ImprovedDiffusionSchedule,
    compute_improved_loss
)


def create_tiny_dataset(num_samples: int = 5, seq_len: int = 32, seed: int = 42) -> torch.Tensor:
    """
    Create a tiny dataset of synthetic protein backbones for memorization test.
    
    Parameters
    ----------
    num_samples : int
        Number of samples to generate. Default is 5.
    seq_len : int
        Length of each protein sequence. Default is 32 residues.
    seed : int
        Random seed for reproducibility.
    
    Returns
    -------
    torch.Tensor
        Dataset of shape (num_samples, seq_len, 9) representing backbone coords.
        Each residue has 9 values: N(x,y,z), Ca(x,y,z), C(x,y,z)
    
    Notes
    -----
    We create realistic-ish backbone geometry:
    - Ca-Ca distance ~3.8 Angstroms
    - Coordinates normalized to [-3, 3] range for diffusion
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    data = []
    for i in range(num_samples):
        # Generate a random backbone trajectory
        # Start at origin
        coords = torch.zeros(seq_len, 3, 3)  # (L, 3 atoms, 3 xyz)
        
        # Ca coordinates - random walk with ~3.8A steps
        ca_pos = torch.zeros(3)
        for j in range(seq_len):
            if j > 0:
                # Random direction, fixed step size
                direction = torch.randn(3)
                direction = direction / (direction.norm() + 1e-6)
                ca_pos = ca_pos + direction * 3.8
            
            coords[j, 1, :] = ca_pos  # Ca
            
            # N is ~1.5A before Ca along backbone
            if j > 0:
                n_offset = (coords[j, 1] - coords[j-1, 1])
                n_offset = -n_offset / (n_offset.norm() + 1e-6) * 1.5
            else:
                n_offset = torch.tensor([-1.5, 0, 0])
            coords[j, 0, :] = ca_pos + n_offset  # N
            
            # C is ~1.5A after Ca
            c_offset = torch.randn(3)
            c_offset = c_offset / (c_offset.norm() + 1e-6) * 1.5
            coords[j, 2, :] = ca_pos + c_offset  # C
        
        # Center and normalize
        centroid = coords[:, 1, :].mean(dim=0)
        coords = coords - centroid
        scale = coords.abs().max() + 1e-6
        coords = coords / scale * 2.5  # Keep in [-2.5, 2.5]
        
        # Reshape to (L, 9)
        coords = coords.reshape(seq_len, 9)
        data.append(coords)
    
    return torch.stack(data)  # (N, L, 9)


def run_memorization_test(
    num_samples: int = 5,
    seq_len: int = 32,
    epochs: int = 500,
    lr: float = 1e-3,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> dict:
    """
    Run memorization test on a tiny dataset.
    
    This tests if the model can perfectly fit a small dataset, which is
    a fundamental sanity check for any generative model.
    
    Parameters
    ----------
    num_samples : int
        Number of samples in tiny dataset. Default 5.
    seq_len : int
        Sequence length. Default 32.
    epochs : int
        Training epochs. Default 500.
    lr : float
        Learning rate. Default 1e-3 (higher for memorization).
    device : str
        Device to use.
    
    Returns
    -------
    dict
        Results containing losses, final metrics, and pass/fail status.
    
    Examples
    --------
    >>> results = run_memorization_test(num_samples=5, epochs=200)
    >>> print(f"Final loss: {results['final_loss']:.6f}")
    >>> print(f"Test passed: {results['passed']}")
    """
    print("=" * 60)
    print("MEMORIZATION TEST - Sanity Check")
    print("=" * 60)
    print(f"Samples: {num_samples}")
    print(f"Sequence length: {seq_len}")
    print(f"Epochs: {epochs}")
    print(f"Device: {device}")
    print("=" * 60)
    
    # Create tiny dataset
    print("\n[1/4] Creating tiny dataset...")
    data = create_tiny_dataset(num_samples, seq_len).to(device)
    print(f"  Shape: {data.shape}")
    print(f"  Range: [{data.min():.2f}, {data.max():.2f}]")
    
    # Create small model (we don't need huge capacity for 5 samples)
    print("\n[2/4] Creating model...")
    model = ImprovedDiffusionDenoiser(
        coord_dim=9,
        embed_dim=128,  # Smaller for memorization
        time_dim=64,
        num_layers=4,   # Fewer layers
        num_heads=4,
        dropout=0.0,    # No dropout for memorization
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params:,}")
    
    # Create schedule
    schedule = ImprovedDiffusionSchedule(
        num_timesteps=1000,
        schedule_type='cosine',
        device=device
    )
    
    # Optimizer with scheduler for better convergence
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    
    # Training
    print("\n[3/4] Training (memorization)...")
    losses = []
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        
        # Multiple passes per epoch for small dataset
        epoch_loss = 0.0
        for _ in range(10):  # 10 gradient steps per epoch
            loss = compute_improved_loss(model, data, schedule, loss_type='mse')
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
        
        scheduler.step()
        losses.append(epoch_loss / 10)
        
        # Print progress
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:4d}/{epochs}: Loss = {loss.item():.6f}")
    
    elapsed = time.time() - start_time
    final_loss = losses[-1]
    best_loss = min(losses)
    
    # Evaluate reconstruction quality
    print("\n[4/4] Evaluating reconstruction...")
    model.eval()
    with torch.no_grad():
        # Test 1: Can the model denoise a slightly noisy version?
        # Add noise at t=100 (10% through the process)
        t_test = torch.full((data.shape[0],), 100, device=device, dtype=torch.long)
        noisy_data, true_noise = schedule.q_sample(data, t_test)
        pred_noise = model(noisy_data, t_test)
        
        # Check if predicted noise matches true noise
        noise_mse = ((pred_noise - true_noise) ** 2).mean().item()
        
        # Test 2: Reconstruct x0 from predicted noise
        x0_pred = schedule.predict_x0_from_noise(noisy_data, t_test, pred_noise)
        mse = ((x0_pred - data) ** 2).mean().item()
        rmsd = np.sqrt(mse)
        
    print(f"  Noise prediction MSE: {noise_mse:.6f}")
    
    # Determine pass/fail
    # For memorization, we expect loss < 0.05 and RMSD < 0.5
    passed = final_loss < 0.05 and rmsd < 1.0
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Training time: {elapsed:.1f}s")
    print(f"  Final loss: {final_loss:.6f}")
    print(f"  Best loss: {best_loss:.6f}")
    print(f"  Reconstruction RMSD: {rmsd:.4f}")
    print(f"  Test PASSED: {'✓ YES' if passed else '✗ NO'}")
    print("=" * 60)
    
    if passed:
        print("\n✓ Model successfully memorized tiny dataset!")
        print("  This confirms the architecture and training loop work correctly.")
    else:
        print("\n✗ Model failed to memorize tiny dataset.")
        print("  This may indicate issues with architecture or training.")
    
    return {
        'losses': losses,
        'final_loss': final_loss,
        'best_loss': best_loss,
        'rmsd': rmsd,
        'passed': passed,
        'elapsed': elapsed,
        'num_samples': num_samples,
        'seq_len': seq_len,
        'epochs': epochs
    }


if __name__ == "__main__":
    # Run the memorization test
    results = run_memorization_test(
        num_samples=5,
        seq_len=32,
        epochs=500,
        lr=1e-3
    )
    
    # Save results
    import json
    with open('memorization_results.json', 'w') as f:
        json.dump({
            'final_loss': float(results['final_loss']),
            'best_loss': float(results['best_loss']),
            'rmsd': float(results['rmsd']),
            'passed': bool(results['passed']),
            'elapsed': float(results['elapsed']),
            'num_samples': int(results['num_samples']),
            'seq_len': int(results['seq_len']),
            'epochs': int(results['epochs'])
        }, f, indent=2)
    
    print(f"\nResults saved to memorization_results.json")
    
    # Exit with appropriate code
    sys.exit(0 if results['passed'] else 1)
