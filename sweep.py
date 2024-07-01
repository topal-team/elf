# wandb agent felisamici/topal-internship/4ha7je2b

import torch
import torch.distributed as dist
from settings import *
from models.simple import SimpleTransformer
from pipeline.partitioners.profile import profile_operations
from pipeline.partitioners.metis import *
from pipeline.partitioners.partition import *
from pipeline import Pipeline
import pickle
#import wandb
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument('--ctype', choices=["rm", "shem"], default="shem")
parser.add_argument('--minconn', type=bool, default=True)
parser.add_argument('--ufactor', type=int, default=30)
parser.add_argument('--niter', type=int, default=10)
parser.add_argument('--ncuts', type=int, default=100)

def profile(model, sample):
    trace = torch.fx.symbolic_trace(model)
    times, memories = profile_operations(trace, sample)
    with open('performance_metrics.pkl', 'wb') as file:
        pickle.dump({'times': times, 'memories': memories}, file)

def evaluate_partition(parts, inputs, sample):
    times = []
    sample = model.get_sample(16)
    x = {i: sample for i in inputs[0]}
    for p in parts:
        start = time.time()
        try:
            x = p(**x)
        except RuntimeError as e:
            print(e)
            return {
                'time': -1,
                'mem': -1
            }
        times.append(time.time() - start)

    total_time = sum(times)

    return total_time

def evaluate_pipeline(parts, sample, placement, schedule):
    pipe = Pipeline(parts, sample.cuda(), placement, schedule = schedule, partition = False)
    # Warmup
    _ = pipe(sample.clone(), torch.empty(0), lambda x,y,**_: x.sum(), sample.size(0) // 8)

    _ = pipe(sample.clone(), torch.empty(0), lambda x,y,**_: x.sum(), sample.size(0) // 8)

    return pipe.times['total']

if __name__ == '__main__':
    rank = int(os.getenv('RANK', 0))
    placement = [0, 1, 2, 3, 0, 1, 2, 3]
    schedule = "1f1b"
    if rank == 0:
        wandb.init(mode='offline')
        args = parser.parse_args()
        # model = SimpleTransformer(256, 64, 16)
        # wandb.config.update(args)
        wandb.config.update({
            'arch': arch,
            'model configuration': conf,
            'batch size': batch_size,
            'sequence length': block_size
        })
        
        trace = torch.fx.symbolic_trace(model)
        n = 4

        with open('performance_metrics.pkl', 'rb') as file:
            data = pickle.load(file)
        times = data['times']
        memories = data['memories']

        graph = convert_fx(trace, times, memories)
        file = write_metis(graph)
        pweights = [getattr(wandb.config, f'uf{i}') for i in range(len(placement))]
        pweights = torch.tensor(pweights, dtype=torch.float32)
        pweights = torch.softmax(pweights, dim=0)
        weights_file = tempfile.NamedTemporaryFile("w+", dir = ".")
        for i, twgt in enumerate(pweights):
            weights_file.write(f'{i} = {twgt}\n')
            wandb.config.update({f'uf{i}': twgt})

        file.flush()  # Ensure all data is written before executing
        file.seek(0)  # Reset file pointer to the beginning for reading in subprocess
        command = ["gpmetis", file.name, str(n), "-objtype=vol", "-contig"]
        command.append(f'-ctype={wandb.config.ctype}')
        command.append(f'-ufactor={wandb.config.ufactor}')
        command.append(f'-niter={wandb.config.niter}')
        command.append(f'-ncuts={wandb.config.ncuts}')
        command.append(f'-tpwgts={weights_file.name}')
        if wandb.config.minconn:
            command.append(f'-minconn')
        subprocess.run(command, stdout=subprocess.DEVNULL)  # Execute gpmetis

        file.close()
        weights_file.close()
        parts = read_metis(graph, f'{file.name}.part.{n}')

        inputs, outputs = get_inputs_outputs(parts)
        mem = sum(sum(memories[n] for n in out) for out in outputs)
        wandb.log({"mem": mem})

        blocks = []
        for i, p in enumerate(parts):
            graph = create_subgraph(trace, p, inputs[i], outputs[i])
            blocks.append(graph)

        partition = list(zip(blocks, inputs.values(), outputs.values()))
        input_list = [[] for _ in range(max(placement) + 1)]
        for i, p in enumerate(placement):
            input_list[p].append(partition[i])
    else:
        input_list = None

    output_list = [None]
    dist.scatter_object_list(output_list, input_list, src = 0)
    model, inputs, outputs = ([m for m, _, _ in output_list[0]],
                              [i for _, i, _ in output_list[0]],
                              [o for _, _, o in output_list[0]])

    # t = evaluate_partition(blocks, inputs, model.get_sample(16))
    t = evaluate_pipeline(blocks, inputs, placement, schedule)
    wandb.log({"time": t})
    
    wandb.finish()
