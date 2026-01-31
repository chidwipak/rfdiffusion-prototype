#!/usr/bin/env python3
"""
Improved Production Training Script for RFDiffusion
====================================================

Fixes from previous training:
1. Uses improved model with proper per-atom noise prediction
2. Cosine schedule instead of linear (better for structures)
3. Higher model capacity (8 transformer layers, 256 dim)
4. Better learning rate schedule with warmup
5. Proper data normalization
6. Gradient clipping and stability improvements

Author: Chidwipak (GSoC 2026)

Expected Results:
- Loss should decrease from ~1.0 to ~0.1 or lower
- This is comparable to DDPM/RFDiffusion papers
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
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import numpy as np

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from improved_diffusion import (
    ImprovedDiffusionDenoiser,
    ImprovedDiffusionSchedule,
    compute_improved_loss
)


class NormalizedCATHDataset(Dataset):
    """
    CATH dataset with proper normalization for diffusion training.
    
    Key improvements:
    1. Center coordinates around origin
    2. Scale to reasonable range
    3. Cache normalized data for efficiency
    """
    
    def __init__(
        self,
        data_dir: str = "./cath_data",
        max_length: int = 128,
        download: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.max_length = max_length
        self.processed_dir = self.data_dir / "processed_tensors"  # Fixed path
        
        if download and not self.processed_dir.exists():
            self._download_and_process()
        
        # Load processed data
        self.data = []
        if self.processed_dir.exists():
            for pt_file in sorted(self.processed_dir.glob("*.pt")):
                try:
                    coords = torch.load(pt_file, weights_only=True)
                    if coords.shape[0] <= max_length:
                        # Normalize the coordinates
                        coords = self._normalize(coords)
                        self.data.append(coords)
                except Exception as e:
                    continue
        
        print(f"[Dataset] Loaded {len(self.data)} proteins (max_length={max_length})")
    
    def _normalize(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Normalize coordinates for diffusion training.
        
        1. Center around Ca centroid
        2. Scale to unit variance
        """
        # coords: (L, 3, 3) -> N, Ca, C atoms
        L = coords.shape[0]
        
        # Get Ca coordinates (middle atom)
        ca_coords = coords[:, 1, :]  # (L, 3)
        
        # Center around Ca centroid
        centroid = ca_coords.mean(dim=0, keepdim=True)
        coords = coords - centroid.unsqueeze(0)
        
        # Scale to reasonable range (typically proteins span ~20-50 Angstroms)
        # We want coordinates roughly in [-3, 3] for stable diffusion
        scale = coords.abs().max() + 1e-6
        coords = coords / scale * 3.0
        
        return coords
    
    def _download_and_process(self):
        """Download and process CATH data."""
        # Import from existing loader
        from cath_loader import CATHDataset as OriginalCATH
        original = OriginalCATH(download=True, max_length=self.max_length)
        # Data is now downloaded and processed
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        coords = self.data[idx]  # (L, 3, 3)
        L = coords.shape[0]
        
        # Pad to max_length
        if L < self.max_length:
            padding = torch.zeros(self.max_length - L, 3, 3)
            coords = torch.cat([coords, padding], dim=0)
        
        # Reshape to (max_length, 9)
        coords = coords.reshape(self.max_length, 9)
        
        return coords


class WarmupCosineScheduler:
    """Learning rate scheduler with warmup and cosine decay."""
    
    def __init__(self, optimizer, warmup_steps: int, total_steps: int, min_lr: float = 1e-6):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.step_count = 0
    
    def step(self):
        self.step_count += 1
        lr_mult = self._get_lr_mult()
        
        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            param_group['lr'] = base_lr * lr_mult
    
    def _get_lr_mult(self):
        if self.step_count < self.warmup_steps:
            # Linear warmup
            return self.step_count / self.warmup_steps
        else:
            # Cosine decay
            progress = (self.step_count - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            return max(self.min_lr / self.base_lrs[0], 0.5 * (1 + np.cos(np.pi * progress)))
    
    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']


class TrainingLogger:
    """Handles all logging operations."""
    
    def __init__(self, log_dir: str, rank: int):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.rank = rank
        
        self.logger = logging.getLogger(f"RFDiffusion-v2-Rank{rank}")
        self.logger.setLevel(logging.INFO)
        
        if rank == 0:
            log_file = self.log_dir / f"training_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            fh = logging.FileHandler(log_file)
            ch = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
            fh.setFormatter(formatter)
            ch.setFormatter(formatter)
            self.logger.addHandler(fh)
            self.logger.addHandler(ch)
            self.log_file = log_file
    
    def info(self, msg: str):
        if self.rank == 0:
            self.logger.info(msg)


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
            "config": {},
            "version": "v2_improved"
        }
        
        self.metrics_file = self.save_dir / "metrics_v2.json"
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
    
    if not dist.is_initialized() and world_size > 1:
        dist.init_process_group(backend="nccl")
    
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    
    return local_rank, world_size, device


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def save_checkpoint(model, optimizer, scheduler, epoch, best_loss, save_dir, is_best=False):
    """Save training checkpoint."""
    model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_step_count': scheduler.step_count,
        'best_loss': best_loss,
    }
    
    torch.save(checkpoint, f"{save_dir}/checkpoint_v2_latest.pt")
    
    if (epoch + 1) % 100 == 0:
        torch.save(checkpoint, f"{save_dir}/checkpoint_v2_epoch_{epoch+1}.pt")
    
    if is_best:
        torch.save(model_state, f"{save_dir}/best_model_v2.pt")
        torch.save(checkpoint, f"{save_dir}/checkpoint_v2_best.pt")


def load_checkpoint(model, optimizer, scheduler, save_dir, device):
    """Load training checkpoint."""
    checkpoint_path = Path(save_dir) / "checkpoint_v2_latest.pt"
    
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        model_state = checkpoint['model_state_dict']
        if hasattr(model, 'module'):
            model.module.load_state_dict(model_state)
        else:
            model.load_state_dict(model_state)
        
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.step_count = checkpoint.get('scheduler_step_count', 0)
        
        return checkpoint['epoch'] + 1, checkpoint['best_loss']
    
    return 0, float('inf')


def train_improved(
    epochs: int = 1000,
    batch_size_per_gpu: int = 2,
    lr: float = 3e-4,
    warmup_epochs: int = 50,
    save_dir: str = "./checkpoints_v2",
    log_dir: str = "./logs_v2",
    embed_dim: int = 256,
    num_layers: int = 8,
    num_heads: int = 8,
    resume: bool = False,
    schedule_type: str = 'cosine',
):
    """
    Improved training with proper diffusion setup.
    
    Expected to achieve loss < 0.1
    """
    
    # Setup DDP
    local_rank, world_size, device = setup_ddp()
    
    # Create directories
    if local_rank == 0:
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
    
    if world_size > 1:
        dist.barrier()
    
    # Initialize logging
    logger = TrainingLogger(log_dir, local_rank)
    metrics = MetricsTracker(save_dir, local_rank)
    
    # Log configuration
    config = {
        "epochs": epochs,
        "batch_size_per_gpu": batch_size_per_gpu,
        "world_size": world_size,
        "effective_batch_size": batch_size_per_gpu * world_size,
        "learning_rate": lr,
        "warmup_epochs": warmup_epochs,
        "embed_dim": embed_dim,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "schedule_type": schedule_type,
        "start_time": datetime.now().isoformat(),
        "model_version": "v2_improved"
    }
    metrics.set_config(config)
    
    logger.info("=" * 70)
    logger.info("RFDIFFUSION IMPROVED TRAINING (v2)")
    logger.info("=" * 70)
    logger.info(f"World Size: {world_size} GPUs")
    logger.info(f"Effective Batch Size: {config['effective_batch_size']}")
    logger.info(f"Model: {embed_dim}d, {num_layers} layers, {num_heads} heads")
    logger.info(f"Schedule: {schedule_type}")
    logger.info(f"Device: {device}")
    logger.info("=" * 70)
    
    # Load dataset with normalization
    logger.info("[1/5] Loading normalized CATH dataset...")
    dataset = NormalizedCATHDataset(download=True, max_length=128)
    
    if len(dataset) == 0:
        logger.info("  [ERROR] No data found!")
        cleanup_ddp()
        return
    
    logger.info(f"  Dataset size: {len(dataset)} proteins")
    
    # Create dataloader
    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=local_rank, shuffle=True)
        loader = DataLoader(dataset, batch_size=batch_size_per_gpu, sampler=sampler, 
                           num_workers=2, pin_memory=True)
    else:
        loader = DataLoader(dataset, batch_size=batch_size_per_gpu, shuffle=True,
                           num_workers=2, pin_memory=True)
        sampler = None
    
    # Create improved model
    logger.info("[2/5] Initializing improved model...")
    model = ImprovedDiffusionDenoiser(
        coord_dim=9,
        embed_dim=embed_dim,
        time_dim=128,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=0.1,
    ).to(device)
    
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Parameters: {num_params:,}")
    
    # Create improved diffusion schedule
    schedule = ImprovedDiffusionSchedule(
        num_timesteps=1000,
        schedule_type=schedule_type,
        device=device
    )
    
    # Optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.99))
    
    # Learning rate scheduler with warmup
    total_steps = epochs * len(loader)
    warmup_steps = warmup_epochs * len(loader)
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps, total_steps)
    
    # Resume from checkpoint
    start_epoch = 0
    best_loss = float('inf')
    
    if resume:
        logger.info("[3/5] Resuming from checkpoint...")
        start_epoch, best_loss = load_checkpoint(model, optimizer, scheduler, save_dir, device)
        if start_epoch > 0:
            logger.info(f"  Resumed from epoch {start_epoch} with best loss {best_loss:.6f}")
        else:
            logger.info("  No checkpoint found, starting fresh")
    else:
        logger.info("[3/5] Starting fresh training...")
    
    # Training loop
    logger.info("[4/5] Starting training loop...")
    logger.info(f"  Epochs: {start_epoch + 1} to {epochs}")
    logger.info(f"  Warmup: {warmup_epochs} epochs")
    logger.info("-" * 70)
    
    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()
        
        if sampler is not None:
            sampler.set_epoch(epoch)
        
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        # Progress bar
        if local_rank == 0:
            pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}", ncols=100)
        else:
            pbar = loader
        
        for batch in pbar:
            x0 = batch.to(device)  # (B, N, 9)
            
            # Compute loss
            loss = compute_improved_loss(model, x0, schedule, loss_type='mse')
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            if local_rank == 0:
                pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_lr():.2e}")
        
        # Average loss
        avg_loss = epoch_loss / num_batches
        
        if world_size > 1:
            loss_tensor = torch.tensor([avg_loss], device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
            avg_loss = loss_tensor.item()
        
        epoch_time = time.time() - epoch_start
        current_lr = scheduler.get_lr()
        
        # Update metrics and save checkpoint
        is_best = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss
        
        if local_rank == 0:
            metrics.update(epoch, avg_loss, current_lr, epoch_time)
            save_checkpoint(model, optimizer, scheduler, epoch, best_loss, save_dir, is_best)
            
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
        logger.info("=" * 70)
        
        # Save final model
        model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
        torch.save(model_state, f"{save_dir}/final_model_v2.pt")
    
    cleanup_ddp()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Improved Training for RFDiffusion (v2)")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--warmup-epochs", type=int, default=50, help="Warmup epochs")
    parser.add_argument("--embed-dim", type=int, default=256, help="Embedding dimension")
    parser.add_argument("--num-layers", type=int, default=8, help="Number of transformer layers")
    parser.add_argument("--num-heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--schedule", type=str, default="cosine", choices=["linear", "cosine"],
                        help="Noise schedule type")
    parser.add_argument("--save-dir", type=str, default="./checkpoints_v2", help="Save directory")
    parser.add_argument("--log-dir", type=str, default="./logs_v2", help="Log directory")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--local-rank", "--local_rank", type=int, default=0)
    
    args = parser.parse_args()
    
    train_improved(
        epochs=args.epochs,
        batch_size_per_gpu=args.batch_size,
        lr=args.lr,
        warmup_epochs=args.warmup_epochs,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        embed_dim=args.embed_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        resume=args.resume,
        schedule_type=args.schedule,
    )
