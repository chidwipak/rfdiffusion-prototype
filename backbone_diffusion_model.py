"""
DeepChem TorchModel Wrapper for Backbone Diffusion

This module wraps the backbone diffusion denoiser for compatibility with
DeepChem's model interface, following the pattern established by SE3TransformerModel.

Author: Chidwipak
Date: January 2026
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Any, List

try:
    from deepchem.models.torch_models import TorchModel
    from deepchem.models.losses import Loss
    DEEPCHEM_AVAILABLE = True
except ImportError:
    DEEPCHEM_AVAILABLE = False
    print("Warning: DeepChem not found. Using standalone mode.")

from backbone_diffusion import (
    BackboneDiffusionDenoiser,
    DiffusionSchedule,
    compute_diffusion_loss
)


class DiffusionLoss(Loss if DEEPCHEM_AVAILABLE else object):
    """
    Custom loss function for diffusion training.
    Wraps the standard DDPM MSE loss.
    """
    
    def __init__(self, schedule: DiffusionSchedule):
        if DEEPCHEM_AVAILABLE:
            super().__init__()
        self.schedule = schedule
    
    def _create_pytorch_loss(self):
        """Required by DeepChem Loss interface."""
        def loss_fn(output, labels):
            # In diffusion, loss is computed differently
            # This is handled in the model's fit method
            return torch.tensor(0.0)
        return loss_fn


class BackboneDiffusionModel(TorchModel if DEEPCHEM_AVAILABLE else nn.Module):
    """
    DeepChem-compatible wrapper for backbone coordinate diffusion.
    
    This model wraps BackboneDiffusionDenoiser following the pattern from
    SE3TransformerModel in deepchem/models/torch_models/se3_transformer.py:
    
    1. __init__: Creates the core model, defines loss, initializes TorchModel
    2. _prepare_batch: Converts DeepChem dataset batch to model inputs
    3. fit: Custom training loop for diffusion
    4. save/reload: Checkpointing methods
    
    Parameters
    ----------
    num_timesteps : int
        Number of diffusion timesteps (default 1000)
    hidden_dim : int
        Hidden dimension of denoiser network (default 256)
    time_dim : int
        Dimension of time embeddings (default 128)
    num_layers : int
        Number of residual blocks (default 4)
    learning_rate : float
        Learning rate for optimizer (default 1e-4)
    device : torch.device, optional
        Device to run on (default auto-detect)
    **kwargs
        Additional arguments for TorchModel
    
    Example
    -------
    >>> import deepchem as dc
    >>> import numpy as np
    >>> # Create dataset with backbone coordinates
    >>> coords = np.random.randn(10, 50, 9).astype(np.float32)  # 10 proteins, 50 residues
    >>> labels = np.zeros((10, 1))  # Dummy labels for unsupervised
    >>> dataset = dc.data.NumpyDataset(X=coords, y=labels)
    >>> # Create and train model
    >>> model = BackboneDiffusionModel(num_timesteps=100, hidden_dim=128)
    >>> loss = model.fit(dataset, nb_epoch=10)
    """
    
    def __init__(
        self,
        num_timesteps: int = 1000,
        hidden_dim: int = 256,
        time_dim: int = 128,
        num_layers: int = 4,
        learning_rate: float = 1e-4,
        device: Optional[torch.device] = None,
        **kwargs
    ):
        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda:0')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = device
        
        # Create diffusion schedule
        self.schedule = DiffusionSchedule(
            num_timesteps=num_timesteps,
            device=self.device
        )
        
        # Create core denoiser model
        model = BackboneDiffusionDenoiser(
            coord_dim=9,  # N, Ca, C backbone atoms
            hidden_dim=hidden_dim,
            time_dim=time_dim,
            num_layers=num_layers
        )
        
        # Store config
        self.num_timesteps = num_timesteps
        self.hidden_dim = hidden_dim
        self.time_dim = time_dim
        self.num_layers = num_layers
        
        if DEEPCHEM_AVAILABLE:
            # Initialize TorchModel
            loss = DiffusionLoss(self.schedule)
            super(BackboneDiffusionModel, self).__init__(
                model,
                loss=loss,
                device=self.device,
                learning_rate=learning_rate,
                **kwargs
            )
        else:
            # Standalone mode
            super().__init__()
            self.model = model.to(self.device)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
    
    def _prepare_batch(
        self,
        batch: Tuple[Any, Any, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare a batch of data for the diffusion model.
        
        Parameters
        ----------
        batch : Tuple[Any, Any, Any]
            A batch from DeepChem dataset (X, y, w)
        
        Returns
        -------
        Tuple
            (coordinates, labels, weights) as tensors
        """
        inputs, labels, weights = batch
        
        # Convert to tensors
        if isinstance(inputs, list):
            inputs = inputs[0]
        
        coords = torch.tensor(inputs, dtype=torch.float32, device=self.device)
        labels = torch.tensor(labels, dtype=torch.float32, device=self.device)
        weights = torch.tensor(weights, dtype=torch.float32, device=self.device)
        
        return coords, labels, weights
    
    def fit(
        self,
        dataset,
        nb_epoch: int = 100,
        batch_size: int = 32,
        verbose: bool = True,
        **kwargs
    ) -> float:
        """
        Train the diffusion model.
        
        Parameters
        ----------
        dataset : dc.data.Dataset
            DeepChem dataset with backbone coordinates
        nb_epoch : int
            Number of training epochs
        batch_size : int
            Batch size for training
        verbose : bool
            Print training progress
        
        Returns
        -------
        float
            Final training loss
        """
        self.model.train()
        
        if DEEPCHEM_AVAILABLE:
            # Use DeepChem's data generator
            from deepchem.data import NumpyDataset
            generator = dataset.iterbatches(
                batch_size=batch_size,
                deterministic=False,
                pad_batches=True
            )
        else:
            # Standalone mode
            generator = self._standalone_batches(dataset, batch_size)
        
        total_loss = 0.0
        num_batches = 0
        
        for epoch in range(nb_epoch):
            epoch_loss = 0.0
            epoch_batches = 0
            
            for batch in dataset.iterbatches(batch_size=batch_size) if DEEPCHEM_AVAILABLE else generator:
                coords, _, _ = self._prepare_batch(batch)
                
                # Compute diffusion loss
                self.optimizer.zero_grad()
                loss = compute_diffusion_loss(self.model, coords, self.schedule)
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                epoch_batches += 1
            
            avg_epoch_loss = epoch_loss / max(epoch_batches, 1)
            total_loss = avg_epoch_loss
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{nb_epoch}, Loss: {avg_epoch_loss:.6f}")
        
        return total_loss
    
    def _standalone_batches(self, data, batch_size):
        """Generate batches in standalone mode."""
        X = data['X'] if isinstance(data, dict) else data
        n_samples = len(X)
        indices = list(range(n_samples))
        
        for i in range(0, n_samples, batch_size):
            batch_idx = indices[i:i+batch_size]
            batch_X = X[batch_idx]
            batch_y = [[0.0]] * len(batch_idx)
            batch_w = [[1.0]] * len(batch_idx)
            yield (batch_X, batch_y, batch_w)
    
    def generate(
        self,
        num_samples: int = 1,
        num_residues: int = 50
    ) -> torch.Tensor:
        """
        Generate new protein backbone coordinates.
        
        Parameters
        ----------
        num_samples : int
            Number of samples to generate
        num_residues : int
            Number of residues per sample
        
        Returns
        -------
        torch.Tensor
            Generated coordinates of shape (num_samples, num_residues, 9)
        """
        self.model.eval()
        with torch.no_grad():
            samples = self.schedule.sample(
                self.model,
                (num_samples, num_residues, 9),
                device=self.device
            )
        return samples
    
    def denoise(
        self,
        noisy_coords: torch.Tensor,
        num_steps: Optional[int] = None
    ) -> torch.Tensor:
        """
        Denoise noisy coordinates.
        
        Parameters
        ----------
        noisy_coords : torch.Tensor
            Noisy coordinates of shape (batch, num_residues, 9)
        num_steps : int, optional
            Number of denoising steps (default: all timesteps)
        
        Returns
        -------
        torch.Tensor
            Denoised coordinates
        """
        if num_steps is None:
            num_steps = self.num_timesteps
        
        self.model.eval()
        x = noisy_coords.to(self.device)
        
        with torch.no_grad():
            for t in reversed(range(num_steps)):
                t_batch = torch.full(
                    (x.shape[0],), t, 
                    device=self.device, 
                    dtype=torch.long
                )
                x = self.schedule.p_sample(self.model, x, t_batch)
        
        return x
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'config': {
                'num_timesteps': self.num_timesteps,
                'hidden_dim': self.hidden_dim,
                'time_dim': self.time_dim,
                'num_layers': self.num_layers
            }
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Checkpoint loaded from {path}")


if __name__ == "__main__":
    print("Testing BackboneDiffusionModel...")
    
    # Test in standalone mode first
    import numpy as np
    
    # Create dummy data
    num_proteins = 20
    num_residues = 30
    coords = np.random.randn(num_proteins, num_residues, 9).astype(np.float32)
    
    if DEEPCHEM_AVAILABLE:
        import deepchem as dc
        labels = np.zeros((num_proteins, 1))
        dataset = dc.data.NumpyDataset(X=coords, y=labels)
        
        # Create model
        model = BackboneDiffusionModel(
            num_timesteps=50,
            hidden_dim=64,
            num_layers=2
        )
        
        # Train
        print("Training...")
        final_loss = model.fit(dataset, nb_epoch=20, batch_size=8)
        print(f"Final loss: {final_loss:.6f}")
        
        # Generate
        print("Generating samples...")
        samples = model.generate(num_samples=2, num_residues=20)
        print(f"Generated shape: {samples.shape}")
        
        # Save/load checkpoint
        model.save_checkpoint("/tmp/test_checkpoint.pt")
        model.load_checkpoint("/tmp/test_checkpoint.pt")
        
        print("All tests passed!")
    else:
        print("DeepChem not available. Skipping integration test.")
        print("Install DeepChem with: pip install deepchem")
