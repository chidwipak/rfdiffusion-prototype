#!/usr/bin/env python3
"""
Multi-GPU Training Script for RFDiffusion Prototype
====================================================

Author: Chidwipak (GSoC 2026)

This script uses PyTorch DistributedDataParallel (DDP) to train on all 4 GPUs.

Usage:
------
# Single command to use all 4 GPUs:
torchrun --nproc_per_node=4 train_multigpu.py --epochs 100

# Or with specific GPUs:
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train_multigpu.py --epochs 100
"""

import os
import sys
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from cath_loader import CATHDataset
from se3_diffusion import SE3DiffusionDenoiser
from backbone_diffusion import DiffusionSchedule


def setup_ddp():
    """Initialize distributed training."""
    # Get local rank from environment
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    # Initialize process group
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    
    # Set device for this process
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    
    return local_rank, world_size, device


def cleanup_ddp():
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def print_rank0(msg, local_rank):
    """Only print on rank 0."""
    if local_rank == 0:
        print(msg)


def train_multigpu(
    epochs: int = 100,
    batch_size_per_gpu: int = 1,
    accum_steps: int = 4,
    lr: float = 1e-4,
    save_dir: str = "./cath_checkpoints_multigpu",
    num_layers: int = 6,
    embed_dim: int = 128
):
    """
    Train on multiple GPUs using DDP.
    
    Parameters
    ----------
    epochs : int
        Number of training epochs.
    batch_size_per_gpu : int
        Batch size per GPU (effective batch = this * num_gpus * accum_steps).
    accum_steps : int
        Gradient accumulation steps.
    lr : float
        Learning rate.
    save_dir : str
        Directory to save checkpoints.
    num_layers : int
        Number of RoseTTAFold layers.
    embed_dim : int
        Embedding dimension.
    """
    
    # Setup DDP
    local_rank, world_size, device = setup_ddp()
    
    print_rank0(f"\n{'='*60}", local_rank)
    print_rank0("MULTI-GPU TRAINING", local_rank)
    print_rank0(f"{'='*60}", local_rank)
    print_rank0(f"World Size: {world_size} GPUs", local_rank)
    print_rank0(f"Effective Batch Size: {batch_size_per_gpu * world_size * accum_steps}", local_rank)
    print_rank0(f"Device: {device}", local_rank)
    print_rank0(f"{'='*60}\n", local_rank)
    
    # Create save directory
    if local_rank == 0:
        os.makedirs(save_dir, exist_ok=True)
    
    # Wait for rank 0 to create directory
    dist.barrier()
    
    # Load dataset (only download on rank 0)
    print_rank0("[1/4] Loading CATH dataset...", local_rank)
    if local_rank == 0:
        dataset = CATHDataset(download=True, max_length=128)
    
    dist.barrier()  # Wait for download
    
    # All ranks load the processed data
    dataset = CATHDataset(download=False, max_length=128)
    
    if len(dataset) == 0:
        print_rank0("[ERROR] No data found!", local_rank)
        cleanup_ddp()
        return
    
    print_rank0(f"  Dataset size: {len(dataset)} proteins", local_rank)
    
    # Create distributed sampler
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=local_rank,
        shuffle=True
    )
    
    # Create dataloader
    loader = DataLoader(
        dataset,
        batch_size=batch_size_per_gpu,
        sampler=sampler,
        num_workers=2,
        pin_memory=True
    )
    
    # Create model
    print_rank0("[2/4] Initializing model...", local_rank)
    model = SE3DiffusionDenoiser(
        embed_dim=embed_dim,
        time_dim=64,
        num_layers=num_layers,
        num_heads=8
    ).to(device)
    
    # Wrap with DDP
    # find_unused_parameters=True needed because some conditioning modules may not be used
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    
    num_params = sum(p.numel() for p in model.parameters())
    print_rank0(f"  Parameters: {num_params:,}", local_rank)
    
    # Optimizer and schedule
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    schedule = DiffusionSchedule(num_timesteps=1000, device=device)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Training loop
    print_rank0("[3/4] Starting training...", local_rank)
    print_rank0(f"  Epochs: {epochs}", local_rank)
    print_rank0(f"  Accumulation Steps: {accum_steps}", local_rank)
    
    best_loss = float('inf')
    
    for epoch in range(epochs):
        # Set epoch for sampler (important for shuffling)
        sampler.set_epoch(epoch)
        
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        start_time = time.time()
        
        # Reset gradients
        optimizer.zero_grad()
        
        # Progress bar only on rank 0
        if local_rank == 0:
            pbar = tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch+1}/{epochs}")
        else:
            pbar = enumerate(loader)
        
        for batch_idx, x0 in pbar:
            x0 = x0.to(device)
            
            # Reshape to (batch, residues, 9)
            batch_size = x0.shape[0]
            x0_flat = x0.reshape(batch_size, -1, 9)
            
            # Sample timesteps
            t = torch.randint(0, schedule.num_timesteps, (batch_size,), device=device)
            
            # Forward diffusion
            noisy_x, noise = schedule.q_sample(x0_flat, t)
            
            # Predict noise
            noise_pred = model(noisy_x, t)
            
            # Compute loss
            loss = nn.functional.mse_loss(noise_pred, noise)
            
            # Scale loss for gradient accumulation
            loss = loss / accum_steps
            loss.backward()
            
            # Accumulate
            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(loader):
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            # Track loss
            epoch_loss += loss.item() * accum_steps
            num_batches += 1
            
            # Update progress bar
            if local_rank == 0:
                pbar.set_postfix(loss=f"{loss.item() * accum_steps:.4f}")
        
        # Step scheduler
        scheduler.step()
        
        # Average loss across all GPUs
        avg_loss = epoch_loss / num_batches
        
        # Reduce loss across all processes
        loss_tensor = torch.tensor([avg_loss], device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
        avg_loss = loss_tensor.item()
        
        elapsed = time.time() - start_time
        
        # Logging (rank 0 only)
        if local_rank == 0:
            print(f"Epoch {epoch+1}: Avg Loss = {avg_loss:.4f} | Time = {elapsed:.1f}s | LR = {scheduler.get_last_lr()[0]:.6f}")
            
            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(
                    model.module.state_dict(),  # .module to get underlying model
                    f"{save_dir}/best_model.pt"
                )
                print(f"  [✓] New best model saved (Loss: {best_loss:.4f})")
            
            # Regular checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                torch.save(
                    model.module.state_dict(),
                    f"{save_dir}/model_epoch_{epoch+1}.pt"
                )
    
    # Final save
    if local_rank == 0:
        torch.save(
            model.module.state_dict(),
            f"{save_dir}/final_model.pt"
        )
        print(f"\n[4/4] Training complete!")
        print(f"  Best Loss: {best_loss:.4f}")
        print(f"  Models saved to: {save_dir}/")
    
    cleanup_ddp()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-GPU Training for RFDiffusion")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size per GPU")
    parser.add_argument("--accum-steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--layers", type=int, default=6, help="Number of RoseTTAFold layers")
    parser.add_argument("--embed-dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--save-dir", type=str, default="./cath_checkpoints_multigpu", 
                        help="Save directory")
    # For torch.distributed.launch compatibility (deprecated but still used)
    parser.add_argument("--local-rank", "--local_rank", type=int, default=0, 
                        help="Local rank for distributed training (set by torch.distributed.launch)")
    
    args = parser.parse_args()
    
    train_multigpu(
        epochs=args.epochs,
        batch_size_per_gpu=args.batch_size,
        accum_steps=args.accum_steps,
        lr=args.lr,
        save_dir=args.save_dir,
        num_layers=args.layers,
        embed_dim=args.embed_dim
    )
