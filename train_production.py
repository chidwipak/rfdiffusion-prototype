#!/usr/bin/env python3
"""
Production Training Script for RFDiffusion Prototype
=====================================================

Author: Chidwipak (GSoC 2026)

Features:
- Multi-GPU training with DistributedDataParallel
- Comprehensive logging to file and console
- Checkpoint saving with resume capability
- Training metrics saved to JSON for analysis
- Robust against SSH disconnections (use with nohup)

Usage:
------
# Full training with logging:
nohup bash -c 'CUDA_VISIBLE_DEVICES=1,2,3 python -m torch.distributed.launch \
    --nproc_per_node=3 --use_env --master_port=29501 \
    train_production.py --epochs 500 --batch-size 1' \
    > logs/train_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Resume from checkpoint:
nohup bash -c 'CUDA_VISIBLE_DEVICES=1,2,3 python -m torch.distributed.launch \
    --nproc_per_node=3 --use_env --master_port=29501 \
    train_production.py --epochs 500 --resume' \
    > logs/train_resume.log 2>&1 &
"""

import os
import sys
import json
import argparse
import time
import logging
from datetime import datetime
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


class TrainingLogger:
    """Handles all logging operations."""
    
    def __init__(self, log_dir: str, rank: int):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.rank = rank
        
        # Setup logging
        self.logger = logging.getLogger(f"RFDiffusion-Rank{rank}")
        self.logger.setLevel(logging.INFO)
        
        if rank == 0:
            # File handler
            log_file = self.log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            fh = logging.FileHandler(log_file)
            fh.setLevel(logging.INFO)
            
            # Console handler
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            
            # Formatter
            formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
            fh.setFormatter(formatter)
            ch.setFormatter(formatter)
            
            self.logger.addHandler(fh)
            self.logger.addHandler(ch)
            
            self.log_file = log_file
            self.logger.info(f"Logging to: {log_file}")
    
    def info(self, msg: str):
        if self.rank == 0:
            self.logger.info(msg)
    
    def warning(self, msg: str):
        if self.rank == 0:
            self.logger.warning(msg)
    
    def error(self, msg: str):
        self.logger.error(f"[Rank {self.rank}] {msg}")


class MetricsTracker:
    """Tracks and saves training metrics."""
    
    def __init__(self, save_dir: str, rank: int):
        self.save_dir = Path(save_dir)
        self.rank = rank
        self.metrics = {
            "train_loss": [],
            "learning_rate": [],
            "epoch_time": [],
            "timestamps": [],
            "best_loss": float('inf'),
            "best_epoch": 0,
            "total_epochs": 0,
            "config": {}
        }
        
        # Load existing metrics if resuming
        self.metrics_file = self.save_dir / "metrics.json"
        if self.metrics_file.exists():
            with open(self.metrics_file, 'r') as f:
                self.metrics = json.load(f)
    
    def update(self, epoch: int, loss: float, lr: float, epoch_time: float):
        if self.rank == 0:
            self.metrics["train_loss"].append(loss)
            self.metrics["learning_rate"].append(lr)
            self.metrics["epoch_time"].append(epoch_time)
            self.metrics["timestamps"].append(datetime.now().isoformat())
            self.metrics["total_epochs"] = epoch + 1
            
            if loss < self.metrics["best_loss"]:
                self.metrics["best_loss"] = loss
                self.metrics["best_epoch"] = epoch + 1
            
            self.save()
    
    def set_config(self, config: dict):
        if self.rank == 0:
            self.metrics["config"] = config
            self.save()
    
    def save(self):
        if self.rank == 0:
            with open(self.metrics_file, 'w') as f:
                json.dump(self.metrics, f, indent=2)
    
    def get_start_epoch(self) -> int:
        return len(self.metrics["train_loss"])


def setup_ddp():
    """Initialize distributed training."""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    
    return local_rank, world_size, device


def cleanup_ddp():
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def save_checkpoint(model, optimizer, scheduler, epoch, best_loss, save_dir, is_best=False):
    """Save training checkpoint with all necessary state."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_loss': best_loss,
    }
    
    # Save latest checkpoint (always)
    torch.save(checkpoint, f"{save_dir}/checkpoint_latest.pt")
    
    # Save epoch checkpoint every 50 epochs
    if (epoch + 1) % 50 == 0:
        torch.save(checkpoint, f"{save_dir}/checkpoint_epoch_{epoch+1}.pt")
    
    # Save best model
    if is_best:
        torch.save(model.module.state_dict(), f"{save_dir}/best_model.pt")
        torch.save(checkpoint, f"{save_dir}/checkpoint_best.pt")


def load_checkpoint(model, optimizer, scheduler, save_dir, device):
    """Load training checkpoint if exists."""
    checkpoint_path = Path(save_dir) / "checkpoint_latest.pt"
    
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint['epoch'] + 1, checkpoint['best_loss']
    
    return 0, float('inf')


def train_production(
    epochs: int = 500,
    batch_size_per_gpu: int = 1,
    accum_steps: int = 4,
    lr: float = 1e-4,
    save_dir: str = "./checkpoints",
    log_dir: str = "./logs",
    num_layers: int = 6,
    embed_dim: int = 128,
    resume: bool = False
):
    """
    Production training with comprehensive logging and checkpointing.
    
    Parameters
    ----------
    epochs : int
        Total number of training epochs.
    batch_size_per_gpu : int
        Batch size per GPU.
    accum_steps : int
        Gradient accumulation steps.
    lr : float
        Initial learning rate.
    save_dir : str
        Directory for checkpoints and models.
    log_dir : str
        Directory for log files.
    num_layers : int
        Number of RoseTTAFold layers.
    embed_dim : int
        Embedding dimension.
    resume : bool
        Whether to resume from last checkpoint.
    """
    
    # Setup DDP
    local_rank, world_size, device = setup_ddp()
    
    # Create directories
    if local_rank == 0:
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
    
    dist.barrier()
    
    # Initialize logging
    logger = TrainingLogger(log_dir, local_rank)
    metrics = MetricsTracker(save_dir, local_rank)
    
    # Log configuration
    config = {
        "epochs": epochs,
        "batch_size_per_gpu": batch_size_per_gpu,
        "world_size": world_size,
        "effective_batch_size": batch_size_per_gpu * world_size * accum_steps,
        "accum_steps": accum_steps,
        "learning_rate": lr,
        "num_layers": num_layers,
        "embed_dim": embed_dim,
        "resume": resume,
        "start_time": datetime.now().isoformat()
    }
    metrics.set_config(config)
    
    logger.info("=" * 70)
    logger.info("RFDIFFUSION PRODUCTION TRAINING")
    logger.info("=" * 70)
    logger.info(f"World Size: {world_size} GPUs")
    logger.info(f"Effective Batch Size: {config['effective_batch_size']}")
    logger.info(f"Device: {device}")
    logger.info(f"Save Directory: {save_dir}")
    logger.info(f"Log Directory: {log_dir}")
    logger.info("=" * 70)
    
    # Load dataset
    logger.info("[1/5] Loading CATH dataset...")
    if local_rank == 0:
        dataset = CATHDataset(download=True, max_length=128)
    
    dist.barrier()
    dataset = CATHDataset(download=False, max_length=128)
    
    if len(dataset) == 0:
        logger.error("No data found!")
        cleanup_ddp()
        return
    
    logger.info(f"  Dataset size: {len(dataset)} proteins")
    
    # Create distributed sampler and dataloader
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=local_rank, shuffle=True)
    loader = DataLoader(dataset, batch_size=batch_size_per_gpu, sampler=sampler, 
                       num_workers=2, pin_memory=True)
    
    # Create model
    logger.info("[2/5] Initializing model...")
    model = SE3DiffusionDenoiser(
        embed_dim=embed_dim,
        time_dim=64,
        num_layers=num_layers,
        num_heads=8
    ).to(device)
    
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Parameters: {num_params:,}")
    
    # Optimizer and schedulers
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    schedule = DiffusionSchedule(num_timesteps=1000, device=device)
    
    # Resume from checkpoint if requested
    start_epoch = 0
    best_loss = float('inf')
    
    if resume:
        logger.info("[3/5] Resuming from checkpoint...")
        start_epoch, best_loss = load_checkpoint(model, optimizer, scheduler, save_dir, device)
        if start_epoch > 0:
            logger.info(f"  Resumed from epoch {start_epoch} with best loss {best_loss:.4f}")
        else:
            logger.info("  No checkpoint found, starting fresh")
    else:
        logger.info("[3/5] Starting fresh training...")
    
    # Training loop
    logger.info("[4/5] Starting training loop...")
    logger.info(f"  Epochs: {start_epoch + 1} to {epochs}")
    logger.info(f"  Accumulation Steps: {accum_steps}")
    logger.info("-" * 70)
    
    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()
        sampler.set_epoch(epoch)
        model.train()
        
        epoch_loss = 0.0
        num_batches = 0
        optimizer.zero_grad()
        
        # Progress bar only on rank 0
        if local_rank == 0:
            pbar = tqdm(enumerate(loader), total=len(loader), 
                       desc=f"Epoch {epoch+1}/{epochs}", ncols=100)
        else:
            pbar = enumerate(loader)
        
        for batch_idx, x0 in pbar:
            x0 = x0.to(device)
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
            loss = loss / accum_steps
            loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            epoch_loss += loss.item() * accum_steps
            num_batches += 1
            
            if local_rank == 0:
                pbar.set_postfix(loss=f"{loss.item() * accum_steps:.4f}")
        
        scheduler.step()
        
        # Average loss across all GPUs
        avg_loss = epoch_loss / num_batches
        loss_tensor = torch.tensor([avg_loss], device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
        avg_loss = loss_tensor.item()
        
        epoch_time = time.time() - epoch_start
        current_lr = scheduler.get_last_lr()[0]
        
        # Update metrics and save checkpoint
        is_best = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss
        
        if local_rank == 0:
            metrics.update(epoch, avg_loss, current_lr, epoch_time)
            save_checkpoint(model, optimizer, scheduler, epoch, best_loss, save_dir, is_best)
            
            # Log progress
            best_marker = " ★ NEW BEST" if is_best else ""
            logger.info(f"Epoch {epoch+1:4d}/{epochs} | Loss: {avg_loss:.6f} | "
                       f"LR: {current_lr:.2e} | Time: {epoch_time:.1f}s{best_marker}")
    
    # Final summary
    if local_rank == 0:
        logger.info("-" * 70)
        logger.info("[5/5] Training Complete!")
        logger.info(f"  Total Epochs: {epochs}")
        logger.info(f"  Best Loss: {best_loss:.6f} (Epoch {metrics.metrics['best_epoch']})")
        logger.info(f"  Models saved to: {save_dir}/")
        logger.info(f"  Metrics saved to: {save_dir}/metrics.json")
        logger.info("=" * 70)
        
        # Save final model
        torch.save(model.module.state_dict(), f"{save_dir}/final_model.pt")
    
    cleanup_ddp()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Production Training for RFDiffusion")
    parser.add_argument("--epochs", type=int, default=500, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size per GPU")
    parser.add_argument("--accum-steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--layers", type=int, default=6, help="Number of RoseTTAFold layers")
    parser.add_argument("--embed-dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--save-dir", type=str, default="./checkpoints", help="Save directory")
    parser.add_argument("--log-dir", type=str, default="./logs", help="Log directory")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--local-rank", "--local_rank", type=int, default=0, 
                        help="Local rank for distributed training")
    
    args = parser.parse_args()
    
    train_production(
        epochs=args.epochs,
        batch_size_per_gpu=args.batch_size,
        accum_steps=args.accum_steps,
        lr=args.lr,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        num_layers=args.layers,
        embed_dim=args.embed_dim,
        resume=args.resume
    )
