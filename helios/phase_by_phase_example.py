import torch
import torch.nn as nn
import torch.distributed as dist
from torchvision.models import resnet50
from pipeline import Pipeline
from pipeline.partitioners.partition import extract_graph, partition_graph
from pipeline.partitioners.profile import profile_operations
import os


if __name__=="__main__":
    # prepare_data on each replica

    # dataset+loader need at each DP replica
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        transforms.Resize((224, 224)),
    ])

    train_dataset = CIFAR10(root='CIFAR10/train',
                            train=True,
                            transform=transform,
                            download=False,
                            )

    test_dataset = CIFAR10(root='CIFAR10/test',
                           train=False,
                           transform=transform,
                           download=False,
                           )

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=config["batch_size"],
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=config["batch_size"])

    # prepare_model: need from one DP replica, 


    model = resnet18(pretrained=False, num_classes=10, zero_init_residual=config["zero_init_residual"])
    sample = None # fx mode: no sample needed # need sample on the same GPU as model

    # build the graph 
    fx_graph = extract_graph(model, sample, mode='fx')

    # profile the graph
    # on one GPU (do we need to consider profiling on several GPU?)
    sample = torch.randn((config["batch_size"], 3, 224, 224))
    times, memories = profile_operations(fx_graph.module(), sample, niter=10)

    # partition
    # actually need split_graph, check_graph, create_subgraphs
    # inside get-inputs-outputs: check for long skip connections
    # blocks - list of modules
    n_parts = 8
    blocks, inputs, outputs = partition_graph(fx_graph.module(), n=n_parts, sample=sample, mode='naive')
    

    # build schedule
    placement = [] # len = num blocks, i = 0, 1, ..., num blocks
    n_mb = 10
    sched_options = {}

    # move to separate function (from pipeline.py)
    shedule_name = '1f1b' # or function
    scheduler_func = _get_scheduler(schedule_name)

    # create an executable schedule
    schedule = scheduler_func(placement, n_mb, sched_options)

    # posprocessing schdule
    # find and fix cycles
    # remove operations belonged to other processes
    # from _generate_function

    
    # run for each process
    # share blocks (scatter within PP, broadcast within DP)
    # objects can be scattered only between GPU
    # done at shared_partition() from pipeline.py

    # initialize pipeline blocks
    # as in create_pipeline in pipeline.py
    # iterate through blocks on each GPU inside 1 process, initialize PipelineBlock
    
    for block in GPU_blocks ...:
        init_pipeline_block()

    # initialize process groups (init_process_group in pipeline.py)
    # 1 PG - per pipeline
    # 1 PG - per DP


    # __call__ from pipeline.py
    # in each process
    # id_to_block from engine will be needed
    # get mb size -> for the train step by engine  (basically for pipeline)
    # train_step - most interesting for profiling -> can get profiling pictures from here
    # with nvtx events for each pipeline block






    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
