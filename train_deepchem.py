#!/usr/bin/env python3
"""
RFDiffusion Training with DeepChem CATH Dataset Integration
============================================================

This training script uses the new DeepChem load_cath() function instead of
the custom CATHDataset loader. This demonstrates proper integration with
DeepChem's data infrastructure.

Author: Chidwipak (GSoC 2026)

Changes from previous version:
- Uses dc.molnet.load_cath() for data loading
- Integrates with DeepChem's Dataset infrastructure
- Maintains same training pipeline and model architecture
"""

import os
import sys
import argparse
import time
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# Add DeepChem to path
sys.path.insert(0, '/home/chidwipak/Gsoc2026/deepchem')

try:
    import deepchem as dc
    print("DeepChem imported successfully")
except ImportError as e:
    print(f"Error importing DeepChem: {e}")
    print("Make sure deepchem is in your Python path")
    sys.exit(1)

# Local imports
sys.path.insert(0, str(Path(__file__).parent))
from improved_diffusion import (
    ImprovedDiffusionDenoiser,
    ImprovedDiffusionSchedule,
    compute_improved_loss
)


class DeepChemDatasetWrapper(torch.utils.data.Dataset):
    """
    Wrapper to use DeepChem Dataset with PyTorch DataLoader.
    
    DeepChem returns datasets with X, y, w, ids attributes.
    This wrapper normalizes the protein coordinates for diffusion training.
    """
    
    def __init__(self, dc_dataset):
        """
        Initialize wrapper.
        
        Parameters
        ----------
        dc_dataset : deepchem.data.Dataset
            DeepChem dataset from load_cath()
        """
        self.dc_dataset = dc_dataset
        self.data = []
        
        # Process and normalize all proteins
        print(f"Processing {len(dc_dataset)} proteins from DeepChem dataset...")
        for i in range(len(dc_dataset)):
            coords = dc_dataset.X[i]
            
            # Skip empty or invalid structures
            if coords.size == 0:
                continue
            
            # Convert to tensor if numpy
            if isinstance(coords, np.ndarray):
                coords = torch.from_numpy(coords).float()
            
            # Normalize coordinates
            coords = self._normalize(coords)
            self.data.append(coords)
        
        print(f"Loaded {len(self.data)} valid proteins after processing")
    
    def _normalize(self, coords):
        """
        Normalize protein coordinates for diffusion training.
        
        Centers around CA centroid and scales to unit variance.
        
        Parameters
        ----------
        coords : torch.Tensor
            Shape (L, 3, 3) where L is sequence length
        
        Returns
        -------
        torch.Tensor
            Normalized coordinates
        """
        # Get CA coordinates (middle atom)
        ca_coords = coords[:, 1, :]
        
        # Center around CA centroid
        centroid = ca_coords.mean(dim=0, keepdim=True)
        coords = coords - centroid.unsqueeze(1)
        
        # Scale to unit variance
        std = coords.reshape(-1, 3).std(dim=0, keepdim=True)
        std = torch.clamp(std, min=1e-6)
        coords = coords / std.unsqueeze(0)
        
        return coords
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


def collate_batch(batch):
    """
    Custom collate function to handle variable length proteins.
    
    Parameters
    ----------
    batch : list of torch.Tensor
        List of protein coordinates
    
    Returns
    -------
    torch.Tensor
        Padded batch of shape (B, max_L, 3, 3)
    """
    max_len = max(x.shape[0] for x in batch)
    batch_size = len(batch)
    
    # Create padded tensor
    padded = torch.zeros(batch_size, max_len, 3, 3)
    
    for i, coords in enumerate(batch):
        L = coords.shape[0]
        padded[i, :L] = coords
    
    return padded


def train_epoch(model, dataloader, optimizer, schedule, device, epoch):
    """
    Train for one epoch.
    
    Parameters
    ----------
    model : nn.Module
        Diffusion denoiser model
    dataloader : DataLoader
        Training data loader
    optimizer : torch.optim.Optimizer
        Optimizer
    schedule : ImprovedDiffusionSchedule
        Diffusion noise schedule
    device : torch.device
        Device to train on
    epoch : int
        Current epoch number
    
    Returns
    -------
    float
        Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, coords in enumerate(pbar):
        coords = coords.to(device)
        B, L = coords.shape[0], coords.shape[1]
        
        # Skip empty batches
        if L == 0:
            continue
        
        # Sample random timesteps
        t = torch.randint(0, schedule.num_steps, (B,), device=device)
        
        # Add noise
        noise = torch.randn_like(coords)
        noisy_coords = schedule.add_noise(coords, t, noise)
        
        # Predict noise
        pred_noise = model(noisy_coords, t)
        
        # Compute loss
        loss = compute_improved_loss(pred_noise, noise, coords)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / max(num_batches, 1)


def main():
    parser = argparse.ArgumentParser(description='Train RFDiffusion with DeepChem integration')
    parser.add_argument('--max-length', type=int, default=128, help='Maximum protein length')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--save-dir', type=str, default='./cath_checkpoints_deepchem', help='Save directory')
    parser.add_argument('--no-split', action='store_true', help='Use full dataset without splitting')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load CATH dataset using DeepChem
    print("\n" + "="*60)
    print("Loading CATH dataset using DeepChem integration")
    print("="*60)
    
    try:
        if args.no_split:
            print("Loading full dataset without splitting...")
            tasks, datasets, transformers = dc.molnet.load_cath(
                featurizer='ProteinBackbone',
                splitter=None,
                max_length=args.max_length,
                reload=True
            )
            dataset = datasets[0]
            print(f"Loaded {len(dataset)} proteins")
        else:
            print("Loading dataset with train/valid/test split...")
            tasks, datasets, transformers = dc.molnet.load_cath(
                featurizer='ProteinBackbone',
                splitter='random',
                max_length=args.max_length,
                reload=True
            )
            train_set, valid_set, test_set = datasets
            print(f"Train: {len(train_set)}, Valid: {len(valid_set)}, Test: {len(test_set)}")
            dataset = train_set
    
    except Exception as e:
        print(f"\nError loading DeepChem dataset: {e}")
        print("\nNote: Make sure you have:")
        print("1. BioPython installed: pip install biopython")
        print("2. Requests installed: pip install requests")
        print("3. Internet connection for downloading PDB files")
        sys.exit(1)
    
    # Wrap DeepChem dataset for PyTorch
    print("\nWrapping DeepChem dataset for PyTorch...")
    torch_dataset = DeepChemDatasetWrapper(dataset)
    
    if len(torch_dataset) == 0:
        print("Error: No valid proteins in dataset")
        sys.exit(1)
    
    # Create DataLoader
    dataloader = DataLoader(
        torch_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
        num_workers=0
    )
    
    # Initialize model and training components
    print("\nInitializing model...")
    model = ImprovedDiffusionDenoiser(
        dim=256,
        num_layers=8,
        num_heads=8
    ).to(device)
    
    schedule = ImprovedDiffusionSchedule(num_steps=1000)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {num_params:,} parameters")
    
    # Training loop
    print("\n" + "="*60)
    print("Starting training")
    print("="*60)
    
    best_loss = float('inf')
    log_file = os.path.join(args.save_dir, 'training_log.txt')
    
    with open(log_file, 'w') as f:
        f.write(f"Training started at {datetime.now()}\n")
        f.write(f"Using DeepChem load_cath() for data loading\n")
        f.write(f"Dataset size: {len(torch_dataset)}\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"Max length: {args.max_length}\n\n")
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        avg_loss = train_epoch(model, dataloader, optimizer, schedule, device, epoch)
        
        epoch_time = time.time() - epoch_start
        
        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"  Loss: {avg_loss:.6f}")
        print(f"  Time: {epoch_time:.2f}s")
        
        # Log to file
        with open(log_file, 'a') as f:
            f.write(f"Epoch {epoch}: loss={avg_loss:.6f}, time={epoch_time:.2f}s\n")
        
        # Save checkpoint
        if epoch % 10 == 0:
            checkpoint_path = os.path.join(args.save_dir, f'model_epoch_{epoch}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"  Saved checkpoint: {checkpoint_path}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = os.path.join(args.save_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, best_path)
            print(f"  New best model! Loss: {avg_loss:.6f}")
    
    print("\n" + "="*60)
    print("Training complete!")
    print(f"Best loss: {best_loss:.6f}")
    print(f"Checkpoints saved to: {args.save_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
