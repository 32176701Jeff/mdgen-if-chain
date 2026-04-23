#
conda create -n mdgen python=3.10 -y
conda activate mdgen

# 
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"

# 
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.1+cu118.html
pip install torch-geometric==2.3.1

#
pip install numpy==1.26.4 pandas==2.3.3 scipy==1.15.3 matplotlib==3.10.8 seaborn==0.13.2 scikit-learn==1.7.2

#
pip install biopython==1.86 fair-esm==2.0.0 pytorch-lightning==2.0.9 torchmetrics==1.8.2 e3nn==0.5.1 einops==0.8.1
pip install wandb==0.23.1 ml-collections==1.1.0 tqdm==4.67.1 pydantic==2.12.5 h5py==3.15.1
python -c "import torch_geometric; import esm; import pytorch_lightning; print('Core packages loaded successfully!')"

#
pip install aiohttp==3.13.2 gitpython==3.1.45 sentry-sdk==2.48.0 torchdiffeq==0.2.5 opt-einsum==3.4.0 dm-tree==0.1.9
pip install mdtraj