import pytest

from elf.partitioners.utils import (
	remove_dupes,
	Signature,
	signatures_from_sources_targets,
	get_sources_targets_sequential,
)


@pytest.mark.unit
def test_remove_dupes():
	seq = [1, 2, 1, 3, 2, 4]
	assert remove_dupes(seq) == [1, 2, 3, 4]


@pytest.mark.unit
def test_signature_methods():
	inputs = ["a", "b"]
	outputs = ["c"]
	sources = [None, 0]
	targets = [[1]]

	sig = Signature(inputs, outputs, sources, targets)

	assert sig.ninputs == 2
	assert sig.noutputs == 1
	assert sig.get_all_sources() == [None, 0]
	assert sig.get_all_targets() == [1]


@pytest.mark.unit
def test_signatures_from_sources_targets():
	# stage 0 : input comes from None, output goes to stage 1
	# stage 1 : input comes from 0, output goes nowhere (None)
	srcs = [{"input": None}, {"x": 0}]
	tgts = [{"output": [1]}, {"output": [None]}]

	sigs = signatures_from_sources_targets(srcs, tgts)
	assert len(sigs) == 2

	s0, s1 = sigs

	assert s0.inputs == ["input"]
	assert s0.outputs == ["output"]
	assert s0.sources == [None]
	assert s0.targets == [[1]]

	assert s1.inputs == ["x"]
	assert s1.outputs == ["output"]
	assert s1.sources == [0]
	assert s1.targets == [[None]]


@pytest.mark.unit
def test_get_sources_targets_sequential():
	placement = [0, 1, 0]
	sources, targets = get_sources_targets_sequential(placement)

	# there should be one entry per stage
	assert len(sources) == len(targets) == 3

	# first stage receives from None, last sends to None
	assert sources[0]["input"] is None
	assert targets[2]["output"] == [None]

	# middle stage plumbing
	assert sources[1]["input"] == 0
	assert targets[0]["output"] == [1]
	assert sources[2]["input"] == 1
	assert targets[1]["output"] == [2]
