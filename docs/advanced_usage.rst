Advanced Usage Guide
====================

This guide covers advanced usage patterns for ELF, including manual partitioning,
custom schedulers, and fine-grained control over pipeline execution.

.. contents:: Table of Contents
   :local:
   :depth: 2


Manual Model Partitioning
--------------------------

While ELF supports automatic partitioning, you can manually partition your model if you want to have control, or if your model can't be traced.

Basic Manual Partitioning
~~~~~~~~~~~~~~~~~~~~~~~~~~

When using manual partitioning, you need to:

1. Split your model into parts yourself
2. Materialize each part to the correct device
3. Define signatures describing the communication pattern
4. Set ``partitioner=False`` in the Pipeline

.. code-block:: python

    import torch
    import torch.nn as nn
    from elf import Pipeline, sequential_signatures

    # Split model manually (example for a sequential model)
    model = YourModel()  # Has .blocks, .embed, .head attributes

    # Determine which blocks go to this rank
    n_blocks = len(model.blocks)
    blocks_per_rank = n_blocks // world_size
    start_idx = rank * blocks_per_rank
    end_idx = (rank + 1) * blocks_per_rank

    # Build the partition for this rank
    if rank == 0:
        # First rank gets embedding + blocks
        part = nn.Sequential(model.embed, *model.blocks[start_idx:end_idx])
    elif rank == world_size - 1:
        # Last rank gets blocks + head
        part = nn.Sequential(*model.blocks[start_idx:end_idx], model.head)
    else:
        # Middle ranks get only blocks
        part = nn.Sequential(*model.blocks[start_idx:end_idx])

    # Build signatures for a sequential pipeline
    placement = list(range(world_size))
    signatures = sequential_signatures(placement)

    # Create pipeline with manual partition
    pipe = Pipeline(
        part,
        partitioner=False,  # Critical: disable automatic partitioning
        placement=placement,
        signatures=signatures,
    )

**Key Points:**

* Set ``partitioner=False`` to skip automatic partitioning
* Provide ``signatures`` describing the communication pattern between stages
* Use ``sequential_signatures(placement)`` for simple sequential models


Signatures for Communication
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Signatures describe the communication pattern between pipeline stages: which stage
sends to which, and what variables are exchanged.

**For Sequential Models:**

Use ``sequential_signatures()`` for models where each stage feeds the next in order:

.. code-block:: python

    from elf import sequential_signatures

    placement = [0, 1, 2, 3]  # Linear placement
    signatures = sequential_signatures(placement)

    # Each signature describes one stage:
    #   signature[0]: input from None (external), output to stage 1
    #   signature[1]: input from stage 0, output to stage 2
    #   ...
    #   signature[N-1]: input from stage N-2, output to None (final result)

**For Non-Sequential Models (Skip Connections):**

For models with skip connections (e.g., ResNets, U-Nets), build ``Signature`` objects
directly:

.. code-block:: python

    from elf.partitioners import Signature

    # Example: 4 stages where stage 2 receives from both stage 0 and stage 1
    signatures = [
        Signature(inputs=["x"],          outputs=["out"],        sources=[None], targets=[[1, 2]]),
        Signature(inputs=["x"],          outputs=["out"],        sources=[0],    targets=[[2]]),
        Signature(inputs=["x", "skip"],  outputs=["out"],        sources=[1, 0], targets=[[3]]),
        Signature(inputs=["x"],          outputs=["out"],        sources=[2],    targets=[[None]]),
    ]

    pipe = Pipeline(
        parts,
        partitioner=False,
        placement=placement,
        signatures=signatures,
    )

**Signature Fields:**

* ``inputs``: variable names this stage receives
* ``outputs``: variable names this stage produces
* ``sources``: for each input, which stage it comes from (``None`` = external input)
* ``targets``: for each output, list of stages it goes to (``None`` = final output)


Data Distribution and Results
------------------------------

Input and Target Placement
~~~~~~~~~~~~~~~~~~~~~~~~~~~

When calling the pipeline, all ranks need to provide the full batch of inputs and targets, because the number of microbatches is inferred from the batch size. All ranks must call the pipeline with the same batch size. Only the data on the rank holding the first (resp. last) stage is used as input (resp. target).

.. code-block:: python

    # All ranks need to provide inputs and targets
    inputs = torch.randn(batch_size, ..., device="cuda")
    targets = torch.randn(batch_size, ..., device="cuda")

    # All ranks call the pipeline
    y, loss = pipe(inputs, targets, loss_fn, split_size=mb_size)


Result Location
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The pipeline results are returned only on the last rank.

.. code-block:: python

    # All ranks provide inputs and targets
    inputs = torch.randn(batch_size, ..., device="cuda")
    targets = torch.randn(batch_size, ..., device="cuda")
    y, loss = pipe(inputs, targets, loss_fn, split_size=mb_size)

    if rank == placement[-1]:
        # Results are available here
        print(f"Output shape: {y.shape}")
        print(f"Loss: {loss.item()}")

        # IMPORTANT: Results are on CPU, not GPU!
        assert y.device.type == "cpu"

        # If you need results on GPU:
        y_gpu = y.cuda()
    else:
        # Other ranks get None
        assert y is None
        assert loss is None

**CPU Offloading**

.. important::
    The output ``y`` is **automatically offloaded to CPU** on the last rank
    for memory savings. This prevents GPU memory from being consumed by result tensors
    that are typically only used for logging or further CPU-side processing (e.g. for validation).

    If you need the results on GPU (e.g., for additional GPU computations), you must
    explicitly move them back with ``.cuda()``.

    An option should be added in the future to disable this behavior.

Placement Patterns
------------------

The ``placement`` parameter defines which device hosts each pipeline stage.
Different schedulers require different placement patterns.

Understanding Placement
~~~~~~~~~~~~~~~~~~~~~~~

A placement is a list where ``placement[stage_id] = device_rank``:

.. code-block:: python

    from elf import Placement

    # Linear: each stage on different device
    placement = Placement([0, 1, 2, 3])

    # Interleaved: multiple stages per device
    placement = Placement([0, 1, 2, 3, 0, 1, 2, 3])
    # Stage 0 and 4 on device 0, stages 1 and 5 on device 1, etc.

    # V-schedule: forward and backward path
    placement = Placement([0, 1, 2, 3, 3, 2, 1, 0])

Automatic Placement Selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``Placement.default()`` to get the appropriate pattern for a scheduler:

.. code-block:: python

    # Automatically selects the right pattern
    placement = Placement.default("1f1b", pp=4)     # [0, 1, 2, 3]
    placement = Placement.default("megatron", pp=4) # [0, 1, 2, 3, 0, 1, 2, 3]
    placement = Placement.default("zbv", pp=4)      # [0, 1, 2, 3, 3, 2, 1, 0]

**Scheduler-Specific Patterns:**

* **1f1b, gpipe, zbh1, zbh2**: Linear ``[0, 1, 2, ...]``
* **megatron**: Interleaved ``[0, 1, 2, ..., 0, 1, 2, ...]`` (2x)
* **hanayo, zbv**: V-schedule ``[0, 1, 2, ..., 2, 1, 0]``


Custom Schedulers
-----------------

You can create custom schedulers to modify execution behavior.

Understanding Schedules
~~~~~~~~~~~~~~~~~~~~~~~~

A schedule is a list of ``Operation`` objects that define what each device does and when:

.. code-block:: python

    from elf.scheduling import Operation, OperationType, OpOptions

    # Example operations in a schedule:
    # Forward pass for block 0, microbatch 0, on rank 0
    op = Operation(block_id=0, mb_id=0, op=OperationType.FORWARD, rank=0)

    # Backward pass for block 1, microbatch 2, on rank 1
    op = Operation(block_id=1, mb_id=2, op=OperationType.BACKWARD_INPUTS, rank=1)

    # Weight gradient computation for block 0, microbatch 0, on rank 0
    op = Operation(block_id=0, mb_id=0, op=OperationType.BACKWARD_PARAMS, rank=0)

**Operation Types:**

* ``FORWARD``: Forward pass through a stage
* ``BACKWARD_INPUTS``: Backward pass (compute input gradients)
* ``BACKWARD_PARAMS``: Weight gradient computation
* ``SEND``, ``RECV``: Communication operations (added automatically)
* ``PREFETCH_ACTIVATIONS``: Prefetch from CPU (for offloading)
* ``RECOMPUTE_FORWARD``, ``RECOMPUTE_BACKWARD_INPUTS``: For checkpointing

Writing a Custom Scheduler
~~~~~~~~~~~~~~~~~~~~~~~~~~~

A scheduler is a function that takes ``(placement, nmb, signatures)`` and returns a schedule:

.. code-block:: python

    from elf.scheduling import Operation, OperationType
    from elf.scheduling.schedulers import (
        _add_forward_pass, _add_backward_pass, _add_backward_params
    )
    from elf.registry import SCHEDULERS

    def my_custom_scheduler(placement, nmb, signatures):
        """
        Custom scheduler that processes microbatches in a specific order.

        Args:
            placement: List[int] - device assignment for each stage
            nmb: int - number of microbatches
            signatures: List[Signature] - input/output info for each stage

        Returns:
            List[Operation] - the execution schedule
        """
        schedule = []
        n_stages = len(placement)
        n_devices = max(placement) + 1

        # Example: All-forward-all-backward (like GPipe/AFAB)
        for rank in range(n_devices):
            # Get stages on this device
            stage_ids = [i for i in range(n_stages) if placement[i] == rank]

            # All forwards
            for mb in range(nmb):
                for stage_id in stage_ids:
                    _add_forward_pass(
                        schedule, placement, stage_id, mb, rank, signatures[stage_id]
                    )

            # All backwards
            for mb in range(nmb):
                for stage_id in reversed(stage_ids):
                    _add_backward_pass(
                        schedule, placement, stage_id, mb, rank, signatures[stage_id]
                    )
                    _add_backward_params(schedule, stage_id, mb, rank)

            # Gradient synchronization (for data parallelism)
            for stage_id in stage_ids:
                schedule.append(
                    Operation(stage_id, None, OperationType.ALL_REDUCE_PARAM_GRADS, rank)
                )

        return schedule

    SCHEDULERS.register("my_custom_scheduler", my_custom_scheduler, "Optional description")
    # Use custom scheduler
    pipe = Pipeline(model, sample, scheduler="my_custom_scheduler")

**Helper Functions:**

ELF provides helper functions to add operations with correct communication:

* ``_add_forward_pass(schedule, placement, block_id, mb_id, rank, signature)``

  Adds RECV (if needed), FORWARD, SEND (if needed), and LOSS_FORWARD (if last stage)

* ``_add_backward_pass(schedule, placement, block_id, mb_id, rank, signature)``

  Adds LOSS_BACKWARD (if last stage), RECV, BACKWARD_INPUTS, and SEND

* ``_add_backward_params(schedule, block_id, mb_id, rank)``

  Adds BACKWARD_PARAMS operation for weight gradient computation

These helpers automatically insert communication operations based on the signatures.

Modifying Existing Schedulers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can wrap existing schedulers to add custom behavior:

.. code-block:: python

    from elf.scheduling import OpOptions, OperationType

    def checkpointed_1f1b(placement, nmb, signatures):
        """1F1B with activation checkpointing"""
        # Get base schedule
        base_scheduler = SCHEDULERS["1f1b"]
        schedule = base_scheduler(placement, nmb, signatures)

        # Modify operations
        for op in schedule:
            if op.op == OperationType.FORWARD:
                # Add checkpointing to all forward passes
                def checkpoint_strategy(name, module):
                    # Checkpoint specific modules
                    return isinstance(module, (nn.GELU, nn.LayerNorm))

                op.options[OpOptions.REMAT_STRATEGY] = checkpoint_strategy

        return schedule

    pipe = Pipeline(model, sample, scheduler=checkpointed_1f1b)

**Available OpOptions:**


See :class:`elf.scheduling.scheduling.OpOptions` for an up-to-date and complete list of available `OpOptions` and their meaning.



Zero Bubble Schedules
---------------------

Zero Bubble (ZB) schedules require special layers that support delayed weight gradient computation.
Only layers that have a ``LayerDW`` equivalent implemented can perform delayed weight gradient computation.

Enabling ZB Schedules
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from elf import replace_linear_with_linear_dw

    model = YourModel()

    # Required before using ZB schedulers
    replace_linear_with_linear_dw(model, device="cuda")

    pipe = Pipeline(model, sample, scheduler="zbh1")  # or "zbh2", "zbv"

**When to use:**

* Only needed for schedulers: ``zbh1``, ``zbh2``, ``zbv``

**Why it's needed:**

ZB schedules compute weight gradients (``BACKWARD_PARAMS``) separately from
the backward pass (``BACKWARD_INPUTS``). This requires buffering intermediate
values, which ``LinearDW`` provides.

.. note::
    If your model contains no ``LayerDW``, then the ``BACKWARD_PARAMS`` operation is a no-op, everything is computed during ``BACKWARD_INPUTS``.


Meta Device with Manual Partitioning
-------------------------------------

For large models, you can create the model on meta device (no memory allocated)
and materialize only the parts needed per rank.

.. code-block:: python

    import torch
    import torch.nn as nn

    # Create model on meta device
    with torch.device("meta"):
        model = YourLargeModel(n_blocks=96)
        replace_linear_with_linear_dw(model, "meta")  # If using ZB

    # Each rank creates only its partition
    # (Model stays on meta device, we just reference its structure)
    n_blocks = len(model.blocks)
    blocks_per_rank = n_blocks // world_size
    start = rank * blocks_per_rank
    end = (rank + 1) * blocks_per_rank

    # Build partition (still on meta)
    if rank == 0:
        part = nn.Sequential(model.embed, *model.blocks[start:end])
    elif rank == world_size - 1:
        part = nn.Sequential(*model.blocks[start:end], model.head)
    else:
        part = nn.Sequential(*model.blocks[start:end])

    # Materialize only this part to device
    part.to_empty(device="cuda")
    for param in part.parameters():
        if hasattr(param, "reset_parameters"):
            param.reset_parameters()

    # Create pipeline
    placement = list(range(world_size))
    signatures = sequential_signatures(placement)
    pipe = Pipeline(
        part, partitioner=False,
        placement=placement,
        signatures=signatures,
    )

**Benefits:**

* Model structure exists on all ranks without using memory
* Only partition assigned to each rank is materialized

For automatic partitioning, this cannot work because we perform an actual forward pass to profile the model.


Loading Schedules from Files
-----------------------------

For advanced optimization, you can compute schedules offline and load them:

.. code-block:: python

    import json
    from elf.registry import SCHEDULERS

    # Load pre-computed schedule
    with open("optimal_schedule.json", "r") as f:
        schedule_dict = json.load(f)

    # Use the fixed scheduler
    scheduler = SCHEDULERS["fixed"](schedule_dict)

    pipe = Pipeline(model, sample, scheduler=scheduler)

The schedule file should be a JSON object with a key `"order"`, whose value is a list of operations.
Each operation should be represented as a tuple:
  (block_id, op_type, mb_id)
where:
  - block_id: integer, the block index
  - op_type: string, either "f" (forward), "b" (backward), or "w" (weight gradient computation)
  - mb_id: integer, the microbatch index

Example `optimal_schedule.json`:
{
    "order": [
        [0, "f", 0],
        [0, "b", 0],
        [0, "w", 0],
        [1, "f", 0],
        [1, "b", 0],
        [1, "w", 0]
    ]
}


Pipeline Execution Behavior
----------------------------

Understanding ELF's Execution Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**GPU Synchronization:**

ELF does NOT synchronize the GPU with the CPU or across devices at the end of
each pipeline iteration. This is by design for performance.

.. code-block:: python

    # This is asynchronous - doesn't wait for GPU completion
    y, loss = pipe(inputs, targets, loss_fn, split_size=mb_size)

    # If you need synchronization (e.g., for timing):
    torch.cuda.synchronize()
    dist.barrier()  # If you need cross-device sync

When running multiple iterations, DON'T synchronize between them:

.. code-block:: python

    # Correct: Let iterations overlap on GPU
    for _ in range(n_iters):
        y, loss = pipe(inputs, targets, loss_fn, split_size=mb_size)
        optimizer.step()

    # Only synchronize at the end if needed
    torch.cuda.synchronize() # GPU-CPU
    dist.barrier() # Between GPUs

Only synchronize when you specifically need it (timing, debugging, etc.).

Pipeline Configuration
~~~~~~~~~~~~~~~~~~~~~~

**PipelineConfig:**

For complex setups, use ``PipelineConfig`` for cleaner code:

.. code-block:: python

    from elf import Pipeline, PipelineConfig, Placement

    config = PipelineConfig(
        scheduler="zbh2",
        placement=Placement.default("zbh2", pp=4),
        partitioner="constrained",  # or False for manual
        dp=2,  # Data parallelism degree
    )

    pipe = Pipeline(model, sample, config=config)

**Data Parallelism:**

Set ``dp > 1`` to use data parallelism with pipeline parallelism:

.. code-block:: python

    # 8 GPUs: 4-way PP, 2-way DP
    config = PipelineConfig(
        scheduler="1f1b",
        pp=4,
        dp=2,
    )

    pipe = Pipeline(model, sample, config=config)

Ranks are organized in a 2D grid: ``[pp_rank, dp_rank]``.
Parameter gradients are all-reduced between replicas at the end of each iteration.

The assumption is that the placement for each pipeline is exactly the same as the first one, shifted by the PP degree.

Advanced Partitioning Control
------------------------------

Partitioner Selection
~~~~~~~~~~~~~~~~~~~~~

ELF provides multiple partitioning algorithms:

.. code-block:: python

    # Naive: very simple load-balancing algorithm, no memory constraints, usually not the best
    pipe = Pipeline(model, sample, partitioner="naive")

    # Constrained: 1 input, 1 output per block (for simple communication) + iterative refinement to improve the partition
    pipe = Pipeline(model, sample, partitioner="constrained")

    # METIS: Graph-based partitioning (requires gpmetis)
    pipe = Pipeline(model, sample, partitioner="metis")

    # dagP: Another graph-based partitioning algorithm (requires rMLGP)
    pipe = Pipeline(model, sample, partitioner="dagP")

    # Manual: Disable automatic partitioning
    pipe = Pipeline(parts, None, partitioner=False, ...)

**Choosing a Partitioner:**

* Start with ``"constrained"`` (default) - good balance
* Use ``"metis"`` or ``"dagP"`` for complex skip connections
* Use ``partitioner=False`` for full manual control

Tracer Selection
~~~~~~~~~~~~~~~~

ELF can use different methods to extract the model graph:

.. code-block:: python

    from elf import PipelineConfig

    # Try all tracers (default)
    config = PipelineConfig(tracer="default")

    # Use specific tracer
    config = PipelineConfig(tracer="fx")         # torch.fx
    config = PipelineConfig(tracer="fx_safe")    # torch.fx with fallbacks
    config = PipelineConfig(tracer="export")     # torch.export

    pipe = Pipeline(model, sample, config=config)

Using torch.export can yield better or worse performance than torch.fx, depending on the model. If both work, try both and compare the performance.
If all tracing algorithms fail, we recommend to use manual partitioning.


Debugging and Introspection
----------------------------

**Pipeline Statistics:**

After execution, access timing information:

.. code-block:: python

    y, loss = pipe(inputs, targets, loss_fn, split_size=mb_size)

    # Access per-rank statistics
    print(pipe.stats)  # Timing info for this rank

**Detailed Profiling:**

.. code-block:: python

    # Enable profiling during execution
    y, loss = pipe(inputs, targets, loss_fn, split_size=mb_size, profile=True)

    # Access detailed statistics
    print(pipe.detailed_stats)

Environment variables ``ELF_TIMINGS`` and/or ``ELF_MEMORY`` need to be set to ``1`` to have meaningful statistics.


Example: Complete Custom Setup
-------------------------------

.. code-block:: python

    import torch
    import torch.nn as nn
    import torch.distributed as dist
    from elf import (
        Pipeline, Placement, replace_linear_with_linear_dw,
        sequential_signatures,
    )
    from elf.scheduling import OpOptions, OperationType
    from elf.registry import SCHEDULERS

    # Initialize distributed
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # 1. Create model on meta device
    with torch.device("meta"):
        model = LargeTransformer(n_blocks=96)
        replace_linear_with_linear_dw(model, "meta")

    # 2. Manual partitioning
    blocks_per_rank = len(model.blocks) // world_size
    start = rank * blocks_per_rank
    end = (rank + 1) * blocks_per_rank

    if rank == 0:
        part = nn.Sequential(model.embed, *model.blocks[start:end])
    elif rank == world_size - 1:
        part = nn.Sequential(*model.blocks[start:end], model.head)
    else:
        part = nn.Sequential(*model.blocks[start:end])

    # Materialize partition
    part.to_empty(device="cuda")
    for param in part.parameters():
        if hasattr(param, "reset_parameters"):
            param.reset_parameters()

    # 3. Custom scheduler with checkpointing
    def my_scheduler(placement, nmb, signatures):
        base = SCHEDULERS["zbh2"]
        schedule = base(placement, nmb, signatures)

        # Add selective checkpointing
        for op in schedule:
            if op.op == OperationType.FORWARD and op.mb_id % 2 == 0:
                # Checkpoint even microbatches
                op.options[OpOptions.REMAT_STRATEGY] = lambda n, m: n == ""

        return schedule

    # 4. Setup communication
    placement = Placement.default("zbh2", world_size)
    signatures = sequential_signatures(placement)

    config = PipelineConfig(
        scheduler=my_scheduler,
        placement=placement,
        partitioner=False,
    )

    # 5. Create pipeline
    pipe = Pipeline(
        part, config=config, signatures=signatures,
    )

    # 6. Training loop
    optimizer = torch.optim.AdamW(pipe.parameters())

    nmb = world_size * 2
    mb_size = 1
    batch_size = nmb * mb_size

    for epoch in range(n_epochs):
        # All ranks need inputs and targets
        inputs = torch.randn(batch_size, seq_len, hidden_dim, device="cuda")
        targets = torch.randn(batch_size, seq_len, hidden_dim, device="cuda")

        optimizer.zero_grad()
        y, loss = pipe(inputs, targets, loss_fn, split_size=mb_size)
        optimizer.step()

        # Results are on CPU on last rank
        if rank == placement[-1]:
            assert y.device.type == "cpu"  # Offloaded to CPU
            print(f"Epoch {epoch}, Loss: {loss.item()}")

        # Note: No synchronization between iterations

    # Only sync at the end if needed
    torch.cuda.synchronize()

    # Cleanup
    pipe.clear()
    dist.destroy_process_group()


See Also
--------

* :doc:`generated/elf.Pipeline`: Main Pipeline API documentation
* :doc:`generated/elf.scheduling`: Scheduling algorithms and operations
* :doc:`generated/elf.partitioners`: Partitioning strategies
