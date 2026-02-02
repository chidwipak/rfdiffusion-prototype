# notes on rfdiffusion paper

read through the watson et al paper to understand what they did

## their approach

trained on whole pdb (~150k structures)
- use 200 timesteps for diffusion
- mse loss between predictions and true structure
- started from pretrained rosettafold weights
- added self conditioning (model sees its own previous predictions)

takes about 2.5 min on rtx a4000 to generate 100 residue protein

they dont actually report training losses in paper, mostly focus on:
- af2 validation scores
- experimental success rates
- rmsd to designs

key thing was fine tuning from RF instead of training from scratch - they mention this was crucial

## my implementation

dataset: 30 cath proteins (thats all i could download/process so far)
- 1000 timesteps
- same mse loss
- trained from scratch (no pretrained weights)
- 256 dim, 8 layers, 8 heads transformer
- 7.5M parameters

hardware: 3x tesla k80s (pretty old gpus)
training: ~12 hours for 1000 epochs

## results comparison

mine:
- loss 0.97 -> 0.033 (96% drop)
- memorization test passes (5 proteins, loss to 0.018)

theirs:
- dont report training loss numbers
- validate with af2, get rmsd 0.9-1.7A on 300-600 residue proteins
- high experimental success

## differences

biggest:
1. dataset size - 30 vs 150k proteins (500x more data)
2. starting point - scratch vs pretrained
3. compute - old k80s vs modern gpus
4. no self conditioning yet (want to add this)

## why the loss looks ok

training from scratch on tiny dataset vs their setup
96% reduction shows its learning the distribution
memorization working means architecture is sound

not trying to match their production system, just showing i understand how it works

## things to add for full replication

- way more data (full pdb)
- self conditioning mechanism  
- pretrained starting point
- better gpus lol

but core diffusion + se3 stuff is working
