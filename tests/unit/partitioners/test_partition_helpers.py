import pytest
import torch.nn as nn
import torch.fx as fx

from elf.partitioners.partition import get_inputs_outputs, get_sources_targets


@pytest.mark.unit
def test_get_inputs_outputs_and_sources_targets():
	model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))
	graph = fx.symbolic_trace(model)

	# Collect nodes by op kind
	placeholder_node = next(n for n in graph.graph.nodes if n.op == "placeholder")
	output_node = next(n for n in graph.graph.nodes if n.op == "output")
	call_nodes = [n for n in graph.graph.nodes if n.op == "call_module"]

	# Partition: [first linear], [relu], [second linear]
	parts = [[placeholder_node, call_nodes[0]], [call_nodes[1]], [call_nodes[2], output_node]]

	inputs, outputs = get_inputs_outputs(parts)

	# Expect:
	# part0 inputs == original placeholder names, outputs some activation name
	assert inputs[0] == [placeholder_node.target]
	assert len(outputs[0]) == 1  # linear output fed to part1
	assert inputs[1] == outputs[0]  # relu takes output of part0
	assert outputs[1]  # relu feeds part2
	assert inputs[2] == outputs[1]

	sources, targets = get_sources_targets(inputs, outputs)

	# For first part, source is None, its single output goes to part1 (index 1)
	assert sources[0] == [None]
	assert targets[0][0] == [1]

	# part1 input source is 0, its output goes to part2
	assert sources[1] == [0]
	assert targets[1][0] == [2]

	# part2 input source is 1, its output target is None (final)
	assert sources[2] == [1]
	assert targets[2][0] == [None]
