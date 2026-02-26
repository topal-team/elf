import pytest


from elf.registry import Registry, resolve


@pytest.mark.unit
def test_registry():
	DummyRegistry: Registry = Registry("dummy")

	def dummy_fn(x):
		return x * 3

	DummyRegistry.register("dummy_fn", dummy_fn, "dummy description")

	# Check that getters work
	assert DummyRegistry["dummy_fn"] == dummy_fn
	assert DummyRegistry.get("dummy_fn") == dummy_fn
	assert DummyRegistry.get_description("dummy_fn") == "dummy description"
	assert "dummy_fn" in DummyRegistry
	assert list(DummyRegistry) == ["dummy_fn"]
	assert DummyRegistry.available() == ["dummy_fn"]

	# Try registering another function with the same name
	with pytest.raises(KeyError):
		DummyRegistry.register("dummy_fn", object())

	# Registering the same function again should work however
	DummyRegistry.register("dummy_fn", dummy_fn)

	# Resolve works both with name and object
	assert resolve("dummy_fn", DummyRegistry) == dummy_fn
	assert resolve(dummy_fn, DummyRegistry) == dummy_fn

	def second_dummy_fn(x):
		return x * 4

	# Register another function, without description
	DummyRegistry.register("second_dummy_fn", second_dummy_fn)

	assert resolve("second_dummy_fn", DummyRegistry) == second_dummy_fn
	assert resolve(second_dummy_fn, DummyRegistry) == second_dummy_fn

	assert list(DummyRegistry) == ["dummy_fn", "second_dummy_fn"]
	assert DummyRegistry.available() == ["dummy_fn", "second_dummy_fn"]

	assert "dummy_fn" in DummyRegistry
	assert "second_dummy_fn" in DummyRegistry

	assert DummyRegistry.get_description("second_dummy_fn") == ""  # no description


@pytest.mark.unit
def test_registry_multiple_aliases():
	"""Test that registering multiple aliases for same object works correctly."""
	reg = Registry("test")

	def my_func(x):
		return x * 2

	reg.register(["alias1", "alias2", "alias3"], my_func)

	assert reg["alias1"] is my_func
	assert reg["alias2"] is my_func
	assert reg["alias3"] is my_func
	assert len(reg) == 3
