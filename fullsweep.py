import wandb
import subprocess
import argparse
import tempfile
import math
import shlex
import os

parser = argparse.ArgumentParser()
parser.add_argument('--minconn', type=bool, default=True)
parser.add_argument('--ufactor', type=int, default=30)
parser.add_argument('--niter', type=int, default=10)
parser.add_argument('--ncuts', type=int, default=100)
for i in range(8):
    parser.add_argument(f'--uf{i}', type=int, default=5)

if __name__ == '__main__':
    args = parser.parse_args()
    wandb.init()

    temperature = 5
    pweights = [getattr(wandb.config, f'uf{i}') / temperature for i in range(8)]
    exp_weights = [math.exp(w) for w in pweights]
    sum_exp_weights = sum(exp_weights)
    pweights = [w / sum_exp_weights for w in exp_weights]
    weights_file = tempfile.NamedTemporaryFile("w+", dir = ".", delete_on_close = False)
    for i,w in enumerate(pweights):
        weights_file.write(f'{i} = {w}\n')
        wandb.config.update({f'uf{i}': w})
    weights_file.flush()
    weights_file.close()
    outfile = tempfile.NamedTemporaryFile("w+", dir = ".")
    command = f"srun --account=hyb@v100 --gpus=4 --nodes=1 --exclusive singularity exec --nv --bind $(pwd -P):$(pwd) {os.environ['SINGULARITY_ALLOWED_DIR']}/pipe.sif torchrun --nproc-per-node=4 --nnodes=1 -- sweep.py --ufactor={wandb.config.ufactor} --niter={wandb.config.niter} --ncuts={wandb.config.ncuts} --wfile={os.path.basename(weights_file.name)} --outfile={os.path.basename(outfile.name)}"
    if wandb.config.minconn:
        command += " --minconn"

    process = subprocess.run(command, shell = True)
    if process.returncode == 0:
        outfile.seek(0)
        output = outfile.read()
        print(f'Output : {output}')
        mem, time, partition_time = tuple(map(float, output.split('\n')))
    else:
        mem = time = partition_time = -1
    
    outfile.close()
    os.remove(weights_file.name)

    weights_file.close()
    wandb.log({
        'time': time,
        'mem': mem,
        'partition_time': partition_time
    })

    wandb.finish()
