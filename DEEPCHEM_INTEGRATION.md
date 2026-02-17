# DeepChem Integration

## Overview

The RFDiffusion model is now properly integrated into DeepChem as a `TorchModel` subclass. All the DeepChem-side code lives in my fork:

**GitHub**: https://github.com/chidwipak/deepchem/tree/gsoc-cath-integration

## What's in the DeepChem Fork

### RFDiffusionModel (deepchem/models/torch_models/rfdiffusion.py)

The main model is `RFDiffusionModel(TorchModel)` — a proper DeepChem TorchModel subclass. It works with standard DeepChem datasets directly:

```python
import deepchem as dc

# load data
tasks, datasets, transformers = dc.molnet.load_cath(
    featurizer='ProteinBackbone', splitter='random')
train, valid, test = datasets

# train - standard deepchem interface
model = dc.models.RFDiffusionModel(
    embed_dim=256, num_layers=8, num_heads=8)
model.fit(train, nb_epoch=100)

# generate new proteins
samples = model.generate(num_samples=5, seq_length=50)
```

No wrappers, no custom data loaders needed. Just `model.fit(dataset)`.

### ProteinBackboneFeaturizer (deepchem/feat/protein_backbone_featurizer.py)

Extracts N/CA/C backbone coordinates from PDB files. Inherits from DeepChem's `Featurizer` base class.

### CATH Dataset Loader (deepchem/molnet/load_function/cath_datasets.py)

MolNet loader following the `load_pdbbind` pattern. Downloads PDB files from RCSB and returns `NumpyDataset` objects.

### Tests

Unit tests for all components in the standard DeepChem test locations.

### Example Script (examples/rfdiffusion_demo.py)

End-to-end demo showing training on CATH and generating new backbones.

## This Repo vs DeepChem Fork

- **This repo** (`rfdiffusion-prototype`): My initial prototype work — experimentation, architecture exploration, training experiments
- **DeepChem fork** (`gsoc-cath-integration` branch): The clean, properly integrated implementation that follows DeepChem patterns
