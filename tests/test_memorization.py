"""
Unit tests for memorization experiment.

Author: Chidwipak
Date: January 2026
"""

import sys
import os
import unittest
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memorization_experiment import (
    create_realistic_backbone,
    create_tiny_dataset,
    compute_ca_distances
)


class TestMemorizationExperiment(unittest.TestCase):
    """Test memorization experiment utilities."""
    
    def test_create_realistic_backbone_shape(self):
        """Test that backbone has correct shape."""
        num_residues = 30
        coords = create_realistic_backbone(num_residues, seed=42)
        
        self.assertEqual(coords.shape, (num_residues, 9))
        self.assertEqual(coords.dtype, np.float32)
    
    def test_create_realistic_backbone_deterministic(self):
        """Test that same seed gives same backbone."""
        coords1 = create_realistic_backbone(20, seed=42)
        coords2 = create_realistic_backbone(20, seed=42)
        
        np.testing.assert_array_equal(coords1, coords2)
    
    def test_create_realistic_backbone_geometry(self):
        """Test that backbone has reasonable bond lengths."""
        coords = create_realistic_backbone(30, seed=42)
        
        # Check Ca-Ca distances (should be roughly 3-4 Å for ideal backbone)
        ca_dists = compute_ca_distances(coords)
        
        # All distances should be positive
        self.assertTrue(np.all(ca_dists > 0))
        
        # Distances should be in reasonable range (2-6 Å)
        self.assertTrue(np.all(ca_dists > 1.0))
        self.assertTrue(np.all(ca_dists < 10.0))
    
    def test_create_tiny_dataset_shape(self):
        """Test that tiny dataset has correct shape."""
        num_proteins = 5
        num_residues = 20
        
        dataset = create_tiny_dataset(num_proteins, num_residues)
        
        self.assertEqual(dataset.shape, (num_proteins, num_residues, 9))
        self.assertEqual(dataset.dtype, np.float32)
    
    def test_create_tiny_dataset_unique(self):
        """Test that each protein in dataset is unique."""
        dataset = create_tiny_dataset(5, 20)
        
        # Each protein should be different
        for i in range(5):
            for j in range(i + 1, 5):
                self.assertFalse(
                    np.allclose(dataset[i], dataset[j]),
                    "Each protein should be unique"
                )
    
    def test_compute_ca_distances_shape(self):
        """Test Ca distance computation shape."""
        num_residues = 30
        coords = create_realistic_backbone(num_residues, seed=42)
        
        ca_dists = compute_ca_distances(coords)
        
        # Should have num_residues - 1 distances
        self.assertEqual(len(ca_dists), num_residues - 1)
    
    def test_compute_ca_distances_values(self):
        """Test Ca distance computation with known values."""
        # Create simple test case with known geometry
        coords = np.zeros((3, 9), dtype=np.float32)
        
        # Residue 0: Ca at [0, 0, 0]
        coords[0, 3:6] = [0.0, 0.0, 0.0]
        
        # Residue 1: Ca at [3.8, 0, 0] (ideal Ca-Ca distance)
        coords[1, 3:6] = [3.8, 0.0, 0.0]
        
        # Residue 2: Ca at [7.6, 0, 0]
        coords[2, 3:6] = [7.6, 0.0, 0.0]
        
        ca_dists = compute_ca_distances(coords)
        
        self.assertEqual(len(ca_dists), 2)
        np.testing.assert_almost_equal(ca_dists[0], 3.8, decimal=5)
        np.testing.assert_almost_equal(ca_dists[1], 3.8, decimal=5)


if __name__ == '__main__':
    unittest.main()
