'''
Load a yaml config and sets all the right variables for benchmark
'''

import torch
import importlib
import yaml

# Load configuration from config.yaml
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

arch = config['model']['arch']
conf = config['model']['conf']
module_name = f"models.{arch}"
module = importlib.import_module(module_name)
ModelClass = getattr(module, arch)
ModelConf = getattr(module, conf)

# Assign configurations to variables
vocab_size = config['model']['vocab_size']
batch_size = config['model']['batch_size']
block_size = config['model']['block_size']
setups = []
options = config['pipeline'].get('options', {})
for name, s in config['pipeline']['setups'].items():
    placement = [int(x) for x in s['placement'].split(',')]
    schedule = s['schedule']
    setups.append((name, placement, schedule))

split_sizes = [int(x) for x in config['pipeline']['split_sizes'].split(',')]
iters = config['iters']
warmups = config['warmups']

inputs = torch.randint(0, vocab_size, (batch_size, block_size))
# TODO: generate these inputs automatically based on the architecture

model = ModelClass(ModelConf(vocab_size, block_size))
