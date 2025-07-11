import os
import sys
import traceback

from datetime import timedelta
from io import StringIO

import torch
import torch.distributed as dist


def init_dist():
	assert "RANK" in os.environ, "Cannot run multi-process tests without torchrun"

	rank = int(os.getenv("RANK"))
	local_rank = int(os.getenv("LOCAL_RANK"))
	world_size = int(os.getenv("WORLD_SIZE"))
	torch.cuda.set_device(local_rank)
	if not dist.is_initialized():
		dist.init_process_group(
			backend="nccl", timeout=timedelta(seconds=60), device_id=torch.device(local_rank)
		)

	return rank, local_rank, world_size


def test(function):
	"""
	Test helper (pytest-like) for distributed tests.
	"""
	rank, local_rank, world_size = init_dist()
	if rank == 0:
		print(f"Testing {function.__name__}...", end="\t")

	# Capture stdout and stderr
	old_stdout = sys.stdout
	old_stderr = sys.stderr
	captured_stdout = StringIO()
	captured_stderr = StringIO()

	try:
		sys.stdout = captured_stdout
		sys.stderr = captured_stderr
		function(rank, local_rank, world_size)
	except Exception as e:
		if rank == 0:
			print("\033[91mFailed\033[0m")
			print(f"Error: {e}")
			print(traceback.format_exc())
	else:
		if rank == 0:
			print("\033[92mPassed\033[0m")
	finally:
		sys.stdout = old_stdout
		sys.stderr = old_stderr

		if rank == 0:
			# Print captured output if any
			stdout_content = captured_stdout.getvalue()
			stderr_content = captured_stderr.getvalue()

			if stdout_content:
				print("Captured stdout:")
				print(stdout_content, end="")

			if stderr_content:
				print("Captured stderr:")
				print(stderr_content, end="")

	if dist.is_initialized():
		dist.destroy_process_group()
