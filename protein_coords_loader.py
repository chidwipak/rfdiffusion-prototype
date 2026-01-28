"""
Protein Backbone Coordinate Loader

Utilities for loading and processing protein backbone coordinates from PDB files.
Extracts N, Ca, C atom coordinates for each residue.

Author: Chidwipak
Date: January 2026
"""

import numpy as np
from typing import List, Tuple, Optional
from pathlib import Path


def parse_pdb_backbone(pdb_path: str) -> np.ndarray:
    """
    Extract backbone atom coordinates (N, Ca, C) from a PDB file.
    
    Parameters
    ----------
    pdb_path : str
        Path to PDB file
    
    Returns
    -------
    np.ndarray
        Backbone coordinates of shape (num_residues, 3, 3)
        where axis 1 is [N, Ca, C] and axis 2 is [x, y, z]
    """
    backbone_atoms = {'N': 0, 'CA': 1, 'C': 2}
    residue_coords = {}  # residue_id -> {atom_name: coords}
    
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith('ATOM'):
                atom_name = line[12:16].strip()
                if atom_name in backbone_atoms:
                    # Parse coordinates
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    
                    # Get residue identifier
                    chain_id = line[21]
                    res_num = int(line[22:26])
                    res_id = (chain_id, res_num)
                    
                    if res_id not in residue_coords:
                        residue_coords[res_id] = {}
                    
                    residue_coords[res_id][atom_name] = np.array([x, y, z])
    
    # Convert to array, only include complete residues
    coords_list = []
    for res_id in sorted(residue_coords.keys()):
        res_atoms = residue_coords[res_id]
        if len(res_atoms) == 3:  # Has all backbone atoms
            coords = np.array([
                res_atoms['N'],
                res_atoms['CA'],
                res_atoms['C']
            ])
            coords_list.append(coords)
    
    if len(coords_list) == 0:
        raise ValueError(f"No complete residues found in {pdb_path}")
    
    return np.array(coords_list, dtype=np.float32)


def normalize_coords(coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Normalize coordinates by centering and scaling.
    
    Parameters
    ----------
    coords : np.ndarray
        Backbone coordinates of shape (num_residues, 3, 3)
    
    Returns
    -------
    Tuple
        (normalized_coords, center, scale)
    """
    # Flatten to (N, 3) for computing statistics
    flat = coords.reshape(-1, 3)
    center = flat.mean(axis=0)
    centered = flat - center
    scale = np.abs(centered).max()
    normalized = centered / scale
    
    return normalized.reshape(coords.shape), center, scale


def flatten_backbone_coords(coords: np.ndarray) -> np.ndarray:
    """
    Flatten backbone coordinates for model input.
    
    Parameters
    ----------
    coords : np.ndarray
        Shape (num_residues, 3, 3) or (batch, num_residues, 3, 3)
    
    Returns
    -------
    np.ndarray
        Shape (num_residues, 9) or (batch, num_residues, 9)
    """
    if coords.ndim == 3:
        return coords.reshape(coords.shape[0], -1)
    elif coords.ndim == 4:
        return coords.reshape(coords.shape[0], coords.shape[1], -1)
    else:
        raise ValueError(f"Expected 3D or 4D array, got {coords.ndim}D")


def unflatten_backbone_coords(flat_coords: np.ndarray) -> np.ndarray:
    """
    Unflatten backbone coordinates back to (N, Ca, C) structure.
    
    Parameters
    ----------
    flat_coords : np.ndarray
        Shape (num_residues, 9) or (batch, num_residues, 9)
    
    Returns
    -------
    np.ndarray
        Shape (num_residues, 3, 3) or (batch, num_residues, 3, 3)
    """
    if flat_coords.ndim == 2:
        return flat_coords.reshape(flat_coords.shape[0], 3, 3)
    elif flat_coords.ndim == 3:
        return flat_coords.reshape(flat_coords.shape[0], flat_coords.shape[1], 3, 3)
    else:
        raise ValueError(f"Expected 2D or 3D array, got {flat_coords.ndim}D")


def load_pdb_as_flat_coords(pdb_path: str, normalize: bool = True) -> np.ndarray:
    """
    Load PDB file and return flattened, normalized backbone coordinates.
    
    Parameters
    ----------
    pdb_path : str
        Path to PDB file
    normalize : bool
        Whether to normalize coordinates
    
    Returns
    -------
    np.ndarray
        Flattened coordinates of shape (num_residues, 9)
    """
    coords = parse_pdb_backbone(pdb_path)
    if normalize:
        coords, _, _ = normalize_coords(coords)
    return flatten_backbone_coords(coords)


def pad_or_truncate(coords: np.ndarray, target_length: int) -> np.ndarray:
    """
    Pad or truncate coordinates to target length.
    
    Parameters
    ----------
    coords : np.ndarray
        Shape (num_residues, 9)
    target_length : int
        Desired number of residues
    
    Returns
    -------
    np.ndarray
        Shape (target_length, 9)
    """
    current_length = coords.shape[0]
    
    if current_length == target_length:
        return coords
    elif current_length > target_length:
        return coords[:target_length]
    else:
        padding = np.zeros((target_length - current_length, 9), dtype=coords.dtype)
        return np.concatenate([coords, padding], axis=0)


def create_diffusion_dataset(
    pdb_files: List[str],
    target_length: int = 50,
    normalize: bool = True
):
    """
    Create a DeepChem NumpyDataset from PDB files.
    
    Parameters
    ----------
    pdb_files : List[str]
        List of paths to PDB files
    target_length : int
        Target number of residues (pad/truncate to this)
    normalize : bool
        Whether to normalize coordinates
    
    Returns
    -------
    dc.data.NumpyDataset or dict
        Dataset with backbone coordinates
    """
    coords_list = []
    valid_files = []
    
    for pdb_path in pdb_files:
        try:
            coords = load_pdb_as_flat_coords(pdb_path, normalize=normalize)
            coords = pad_or_truncate(coords, target_length)
            coords_list.append(coords)
            valid_files.append(pdb_path)
        except Exception as e:
            print(f"Warning: Could not load {pdb_path}: {e}")
    
    if len(coords_list) == 0:
        raise ValueError("No valid PDB files found")
    
    X = np.array(coords_list, dtype=np.float32)
    y = np.zeros((len(coords_list), 1), dtype=np.float32)  # Dummy labels
    
    try:
        import deepchem as dc
        return dc.data.NumpyDataset(X=X, y=y)
    except ImportError:
        return {'X': X, 'y': y}


def find_pdb_files(directory: str) -> List[str]:
    """
    Find all PDB files in a directory.
    
    Parameters
    ----------
    directory : str
        Path to directory
    
    Returns
    -------
    List[str]
        List of PDB file paths
    """
    path = Path(directory)
    return [str(f) for f in path.glob("*.pdb")]


if __name__ == "__main__":
    import sys
    
    # Test with DeepChem test data
    test_pdb = "/home/chidwipak/Gsoc2026/deepchem/deepchem/feat/tests/data/3zso_protein.pdb"
    
    print("Testing protein coordinate loader...")
    
    try:
        # Test parsing
        coords = parse_pdb_backbone(test_pdb)
        print(f"Parsed backbone coords shape: {coords.shape}")
        print(f"  - {coords.shape[0]} residues")
        print(f"  - 3 backbone atoms (N, Ca, C)")
        print(f"  - 3 coordinates (x, y, z)")
        
        # Test normalization
        norm_coords, center, scale = normalize_coords(coords)
        print(f"Normalized coords range: [{norm_coords.min():.3f}, {norm_coords.max():.3f}]")
        
        # Test flattening
        flat_coords = flatten_backbone_coords(coords)
        print(f"Flattened shape: {flat_coords.shape}")
        
        # Test unflattening
        unflat = unflatten_backbone_coords(flat_coords)
        assert np.allclose(coords, unflat), "Unflatten failed!"
        print("Flatten/unflatten roundtrip: OK")
        
        # Test padding
        padded = pad_or_truncate(flat_coords, 100)
        print(f"Padded shape: {padded.shape}")
        
        # Test dataset creation
        dataset = create_diffusion_dataset([test_pdb], target_length=50)
        if isinstance(dataset, dict):
            print(f"Dataset X shape: {dataset['X'].shape}")
        else:
            print(f"Dataset X shape: {dataset.X.shape}")
        
        print("\nAll tests passed!")
        
    except FileNotFoundError:
        print(f"Test PDB file not found: {test_pdb}")
        print("Make sure DeepChem is cloned in the expected location.")
        sys.exit(1)
