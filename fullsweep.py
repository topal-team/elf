import wandb
import subprocess
import argparse
import tempfile
from settings import *

parser = argparse.ArgumentParser()
parser.add_argument('--minconn', type=bool, default=True)
parser.add_argument('--ufactor', type=int, default=30)
parser.add_argument('--niter', type=int, default=10)
parser.add_argument('--ncuts', type=int, default=100)

if __name__ == '__main__':
    args = parser.parse_args()
    wandb.init(dir = ".")
    wandb.config.update({
            'arch': arch,
            'model configuration': conf,
            'batch size': batch_size,
            'sequence length': block_size
        })
    
    pweights = [getattr(wandb.config, f'uf{i}') for i in range(len(placement))]
    pweights = torch.tensor(pweights, dtype=torch.float32)
    pweights = torch.softmax(pweights, dim=0)
    weights_file = tempfile.NamedTemporaryFile("w+", dir = ".")
    for i,w in enumerate(pweights):
        weights_file.write(f'{i} = {w}\n')
        wandb.config.update({f'uf{i}': w})
    
    output = subprocess.getoutput(["sbatch", "--account=hyb@v100", "--gpus=4", "--nodes=1", "--exclusive",
                    "singularity", "exev", "--nv", "--bind", "$(pwd):$(pwd)", "$SINGULARITY_ALLOWED_DIR/pipe.sif",
                    "torchrun" "--nproc-per-node=4", "--nnodes=1", "--", 
                    "sweep.py", f"--ufactor={wandb.config.ufactor}", f"--niter={wandb.config.niter}", f"--ncuts={wandb.config.ncuts}",
                    f"--wfile={weights_file.name}", "--minconn" if wandb.config.minconn else ""
                    ])

    weights_file.close()
    mem, time = tuple(map(float, output.split('\n')))
    wandb.log({
        'time': time,
        'mem': mem
    })

    wandb.finish()