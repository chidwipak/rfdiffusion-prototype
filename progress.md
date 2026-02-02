# rfdiffusion implementation progress

## feb 2 update - major progress

just finished a big training run and got some solid results

### what i did this week

spent last 2 days debugging and training after bharath suggested the memorization test

1. **memorization test** - tried training on just 5 proteins
   - wanted to see if model can actually learn anything
   - trained for 500 epochs
   - loss went from 1.0 down to 0.018
   - rmsd 0.025A 
   - basically proves the architecture works

2. **full cath training** - scaled up to multi gpu
   - got 3 tesla k80s working with ddp
   - trained on ~30 cath proteins
   - 1000 epochs took about 12 hours
   - loss: 0.97 -> 0.033 (96% reduction)
   
3. **fixed a bunch of stuff**
   - k80s dont support some cudnn operations so had to replace conv2d with linear
   - view() was breaking, changed to reshape()
   - added per-atom noise prediction (was broadcasting before which was wrong)

### architecture details

ended up going with transformer instead of full rosettafold because:
- easier to debug
- still captures the important stuff
- 7.5M parameters
- 256 dim embeddings, 8 layers, 8 heads

using mse loss same as the paper
cosine noise schedule with 1000 timesteps

### code quality stuff

bharath wanted:
- numpy docstrings - added to all modules
- type annotations - put them everywhere
- unit tests - wrote 38, all passing
- memorization test - done, works

### comparing to paper

read through watson et al carefully

they have:
- 150k pdb structures (i have 30)
- pretrained rosettafold to start (im from scratch)
- self conditioning (havent added yet)
- modern gpus (i have k80s from like 2014)

but core mechanism is same - se3 diffusion with mse loss

### whats next

if i get gsoc:
- get full pdb dataset access
- add self conditioning 
- integrate with deepchem
- train properly on good hardware

## earlier work

### v1 - basic diffusion
- simple mlp denoiser
- worked but not great
- didnt understand geometry

### v2 - se3 layers
- added invariant point attention
- rotation equivariant
- way better structure quality

### v3 - current transformer model
- per atom noise
- multi gpu support
- actually converges well

## files

main code:
- improved_diffusion.py
- train_improved.py  
- run_memorization_test.py
- se3_diffusion.py (older version)
- rosettafold.py (tried this but went simpler)

tests:
- tests/ has 38 unit tests
- test_memorization.py for the sanity check
- all passing rn

results:
- checkpoints_v2/metrics_v2.json has full training history
- memorization_results.json has the tiny dataset results
