import sys
import torch
import torch.distributed as dist
from settings import *
from pipeline.partitioners.profile import profile_operations
from pipeline.partitioners.metis import *
from pipeline.partitioners.partition import *
from pipeline import Pipeline
import pickle
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument('--ufactor', type=int, default=30)
parser.add_argument('--niter', type=int, default=10)
parser.add_argument('--ncuts', type=int, default=100)
parser.add_argument('--wfile', type=str)
parser.add_argument('--outfile', type=str)

def profile(trace, sample):
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
            print(e, file = sys.stderr)
            return -1
        times.append(time.time() - start)

    total_time = sum(times)

    return total_time

def evaluate_pipeline(parts, sample, placement, schedule):
    sample = sample.cuda()
    
    pipe = Pipeline(parts, sample, placement, schedule = schedule, partition = False)
    
    # Warmup
    _ = pipe(sample.clone(), torch.empty(0), lambda x,y,**_: x.sum(), sample.size(0) // 8)

    _ = pipe(sample.clone(), torch.empty(0), lambda x,y,**_: x.sum(), sample.size(0) // 8)

    return pipe.times['total']

if __name__ == "__main__":
    rank = int(os.getenv('RANK', 0))
    local_rank = int(os.getenv('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    sample = inputs
    args = parser.parse_args()
    
    placement = [0, 1, 2, 3, 0, 1, 2, 3]
    schedule = "1f1b"
    if rank == 0:
        trace = torch.fx.symbolic_trace(model)
        n = len(placement)

        if not os.path.exists('performance_metrics.pkl'):
            profile(trace, inputs)
        with open('performance_metrics.pkl', 'rb') as file:
            data = pickle.load(file)
        times = data['times']
        memories = data['memories']

        graph = convert_fx(trace, times, memories)
        file = write_metis(graph)
        
        file.flush()  # Ensure all data is written before executing
        file.seek(0)  # Reset file pointer to the beginning for reading in subprocess
        command = ["gpmetis", file.name, str(n), "-objtype=vol", "-contig", "-minconn"]
        command.append(f'-ufactor={args.ufactor}')
        command.append(f'-niter={args.niter}')
        command.append(f'-ncuts={args.ncuts}')
        command.append(f'-tpwgts={args.wfile}')
        
        start = time.time()
        subprocess.run(command)  # Execute gpmetis
        partition_time = time.time() - start

        file.close()
        parts = read_metis(graph, f'{file.name}.part.{n}')
        
        while len(parts) != n:
            parts.append([])

        inputs, outputs = get_inputs_outputs(parts)
        mem = sum(sum(memories[n] for n in out) for out in outputs.values())

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
    t = evaluate_pipeline(model, sample, placement, schedule)
    if dist.is_initialized():
        dist.destroy_process_group()
        
    if rank == 0:
        with open(args.outfile, "wb") as f:
            pickle.dump({
                'time': t,
                'mem': mem,
                'partition_time': partition_time
            }, f)
