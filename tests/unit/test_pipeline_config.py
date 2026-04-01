import pytest

from elf.pipeline import PipelineConfig


@pytest.mark.unit
def test_pipeline_config_defaults():
	config = PipelineConfig()

	assert config.tracer == "default"
	assert config.partitioner == "constrained"
	assert config.scheduler == "auto"
	assert config.placement == "auto"
	assert config.pp is None
	assert config.dp == 1
	assert config.worker == 0


@pytest.mark.unit
def test_pipeline_config_custom():
	config = PipelineConfig(
		tracer="fx", partitioner="metis", scheduler="afab", placement="manual", pp=4, dp=2, worker=1
	)

	assert config.tracer == "fx"
	assert config.partitioner == "metis"
	assert config.scheduler == "afab"
	assert config.placement == "manual"
	assert config.pp == 4
	assert config.dp == 2
	assert config.worker == 1


@pytest.mark.unit
def test_pipeline_config_to_kwargs():
	config = PipelineConfig(pp=4, dp=2)
	kwargs = config.to_kwargs()

	assert "tracer" in kwargs
	assert "partitioner" in kwargs
	assert "scheduler" in kwargs
	assert "placement" in kwargs
	assert "pp" in kwargs
	assert "dp" in kwargs
	assert "worker" in kwargs

	assert kwargs["pp"] == 4
	assert kwargs["dp"] == 2
	assert kwargs["tracer"] == "default"


@pytest.mark.unit
def test_pipeline_config_partial():
	config = PipelineConfig(scheduler="zbh2", dp=4)

	assert config.scheduler == "zbh2"
	assert config.dp == 4
	assert config.tracer == "default"
	assert config.partitioner == "constrained"
