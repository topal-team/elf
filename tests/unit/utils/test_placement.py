import pytest

from elf.utils import Placement


@pytest.mark.unit
def test_placement_get_ids():
	placement = Placement([0, 1, 0, 2])
	assert placement.get_ids(0) == [0, 2]
	assert placement.get_ids(1) == [1]
	assert placement.get_ids(2) == [3]


@pytest.mark.unit
def test_placement_default():
	# Default for unknown scheduler is sequential list
	assert Placement.default("afab", 3) == [0, 1, 2]

	# Hanayo / zbv pattern duplicates then reverses
	assert Placement.default("hanayo", 2) == [0, 1, 1, 0]


@pytest.mark.unit
def test_placement_default_megatron():
	placement = Placement.default("megatron", 4)
	assert placement == [0, 1, 2, 3, 0, 1, 2, 3]


@pytest.mark.unit
def test_placement_v_schedule():
	placement = Placement([0, 1, 2, 3, 3, 2, 1, 0])
	assert placement.get_ids(0) == [0, 7]
	assert placement.get_ids(3) == [3, 4]
