import torch
import torch.nn as nn

import rockmate
import rkgb


def sequential_tfno(model, devices, dtype=torch.float32):
	"""
	Returns an ``nn.Sequential`` of TFNO model balanced across the given devices.
	N.B.: this function does not dedup devices.
	"""
	# put all layers into a list
	lifting = model.lifting
	blocks = [model.fno_blocks[i] for i in range(model.n_layers)]
	projection = model.projection

	layers = [lifting, *blocks, projection]

	# partition layers into the given devices
	def numel(layer):
		return sum([p.numel() for p in layer.parameters()])

	total_numel = sum([numel(layer) for layer in layers])
	print("total numel", total_numel)
	phase_numel = total_numel // len(devices)
	delim_numel = phase_numel
	accum_numel = 0

	# seal one pipeline phase when its numel is larger than phase_numel
	phases = [[]]
	for layer in layers:
		phases[-1].append(layer)
		accum_numel += numel(layer)
		if accum_numel > delim_numel:
			delim_numel += phase_numel
			phases.append([])

	# pack all remaining layers into the last phase
	while len(phases) > len(devices):
		phases[-2].extend(phases[-1])
		phases.pop()

	for i, phase in enumerate(phases):
		for layer in phase:
			layer.to(device=torch.device(devices[i]))
		# break

	# create nn.Sequential
	return nn.Sequential(*[nn.Sequential(*phase) for phase in phases])


def sequential_fno(model, devices, dtype=torch.float32):
	"""
	Returns an ``nn.Sequential`` of FNO models balanced across the given devices.
	N.B.: this function does not dedup devices.
	"""

	layers = [layer for layer in model.children()]
	# print(layers)

	# partition layers into the given devices
	def numel(layer):
		return sum([p.numel() for p in layer.parameters()])

	total_numel = sum([numel(layer) for layer in layers])
	# print("total numel", total_numel)
	phase_numel = total_numel // len(devices)
	delim_numel = phase_numel
	accum_numel = 0

	# seal one pipeline phase when its numel is larger than phase_numel
	phases = [[]]
	for layer in layers:
		phases[-1].append(layer)
		accum_numel += numel(layer)
		if accum_numel > delim_numel:
			delim_numel += phase_numel
			phases.append([])

	# pack all remaining layers into the last phase
	while len(phases) > len(devices):
		phases[-2].extend(phases[-1])
		phases.pop()

	for i, phase in enumerate(phases):
		# print(phase)
		for layer in phase:
			layer.to(device=torch.device(devices[i]))
		# break

	# create nn.Sequential
	return nn.Sequential(*[nn.Sequential(*phase) for phase in phases])


class PipelineFNO(nn.Sequential):
	def __init__(self, model, devices, microbatch_size, input_shape, checkpoint=False, budget=None):
		super(PipelineFNO, self).__init__()
		# operator = FNO(n_modes=(16, 16), hidden_channels=64, in_channels=3, out_channels=1)
		self.sequential_model = sequential_fno(model, devices)
		self.microbatch_size = microbatch_size
		rkMods = []

		if checkpoint and budget is not None:
			for module in self.sequential_model.children():
				device = next(module.named_parameters())[1].device
				input = torch.randn(microbatch_size, *input_shape[1:], requires_grad=True)

				# list_solver = [rockmate.solvers.HILP()]
				list_solver = [rockmate.solvers.TwRemat()]
				max_size_S_graph_for_no_partitioning = 0
				partitioners = [rkgb.Ptools.Partitioner_seq(sub_partitioner=rkgb.Ptools.Partitioner())]

				input = input.to(device)
				rkMod = rockmate.HRockmate(
					module,
					input,
					budget,
					list_solvers=list_solver,
					partitioners=partitioners,
					# solve_sched = False,
					max_size_S_graph_for_no_partitioning=max_size_S_graph_for_no_partitioning,
				)

				with torch.no_grad():
					output = module(input)
					input_shape = output.shape

				torch.cuda.empty_cache()

				rkMod.solve_sched(budget, rec=False)
				rkMod.get_compiled_fct()

				rkMods.append(rkMod)

			del input

			self.sequential_model = nn.Sequential(*rkMods)
			print(self.sequential_model)

	def forward(self, x):
		for module in self.sequential_model.children():
			device = next(module.named_parameters())[1].device
			print("forward", module, device)
			x = x.to(device)
			x = module(x)

		return x


# input_shape = (16, 64, 1)
# devices = [f"cuda:{d}" for d in range(torch.cuda.device_count())]
# # tfno_seq = sequential_tfno(operator, devices)
# # tfno_seq_model = SequentialTFNO(devices)
# # sequential_model = sequential_fno([16], 64, 4, devices)
# # print(sequential_model)

# fno1d = FNO1d.FNO1d(16, 64, 4)
# microbatch_size = 16
# fno_seq_model = PipelineFNO(fno1d, devices, microbatch_size, input_shape, checkpoint=True, budget = 15000 * 1024 ** 2)
# print(fno_seq_model)
# inputs = torch.rand(*input_shape, requires_grad=True)
# from copy import deepcopy
# copy_inputs = deepcopy(inputs)

# sequential_outputs = fno_seq_model(inputs)
# print(sequential_outputs.device)
# sequential_outputs.sum().backward()

# copy_inputs = copy_inputs.to('cuda:1')
# origin_outputs = fno1d.to('cuda:1')(copy_inputs)
# origin_outputs.sum().backward()

# print(torch.allclose(sequential_outputs, origin_outputs))
