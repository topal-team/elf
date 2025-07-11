import pytest

from elf.utils import NameMapping


@pytest.mark.unit
def test_name_mapping_roundtrip():
	nm = NameMapping(["in1", "in2"], ["out1", "out2"])

	# forward mapping
	assert nm.to_output("in1") == "out1"
	assert nm.to_output("in2") == "out2"

	# reverse mapping
	assert nm.to_input("out1") == "in1"
	assert nm.to_input("out2") == "in2"

	# unknown keys should raise KeyError
	with pytest.raises(KeyError):
		_ = nm.to_output("does_not_exist")

	with pytest.raises(KeyError):
		_ = nm.to_input("does_not_exist")
