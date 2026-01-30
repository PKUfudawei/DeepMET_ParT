# DeepMET with particle transformer

# Dataset
NanoAOD root files of DYJetsToMuMu and TTTo2L2Nu: [DeepMET](https://huggingface.co/datasets/delvee/DeepMET/tree/main)

# Dependecies
`pip3 install -r requirements.txt`

# Preprocess root to hdf5
`python3 utils/prepare_hdf5.py`

# Train on hdf5 files
Check the parser arguments setup before you run.

### Mutltiple GPUs
`torchrun --nproc-per-node=4 train.py -g 4,5,6,7`

### Single GPU
`python3 train.py -g 0`
