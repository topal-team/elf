#!/bin/bash -l
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=72
#SBATCH --time=00:10:00
#SBATCH --output=out_files/out.out
#SBATCH --error=out_files/err.err
#SBATCH --gres=gpu:4
#SBATCH --account=tutorial
 
# IMPORTANT: load the modules for machine learning tasks and libraries
module load ML-bundle/24.06a

export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0,1,2,3

cd $SCRATCH
 
# create and activate the virtual environment
# python3 -m venv $SCRATCH/venvs/test_env
source $SCRATCH/venvs/test_env/bin/activate
echo "Python environment has been activated" 

# Check for NVIDIA GPUs using nvidia-smi
if command -v nvidia-smi &> /dev/null
then
    echo "NVIDIA GPU found:"
    nvidia-smi
else
    echo "No NVIDIA GPU found."
fi

echo "python3 --version"
# install one of torch versions available at Helios wheel repo
python3 -m pip install --no-cache-dir torch==2.4.0+cu124.post2 torchvision==0.19.0+cu124 
echo "Torch has been installed" 

python3 -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.is_available())"


torchrun --nnodes=1 --nproc_per_node=4 $HOME/topal-internship/example.py

