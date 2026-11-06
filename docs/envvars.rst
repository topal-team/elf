Environment Variables
=====================


ELF_TIMINGS
------------

If set to 1, ELF will insert synchronization points to accurately measure the timing of each operation during execution. The resulting values are returned in the ``detailed_stats`` field of the ``Pipeline`` object after an iteration. Be careful, this may affect performance. If this value is not set, the timing values are meaningless.

ELF_MEMORY
----------

If set to 1, ELF will insert synchronization points to accurately measure the memory usage of each operation during execution. The resulting values are returned in the ``detailed_stats`` field of the ``Pipeline`` object after an iteration. Be careful, this may affect performance. If this value is not set, the memory values are meaningless.

ELF_TIMEOUT
-----------

The timeout for the NCCL watchdog, in seconds. If the process is running for too long, the NCCL watchdog will kill the process. This is useful to avoid hanging processes in case of a bug in the code.

ELF_ILP_SOLVER
--------------

The name of the solver to use for the ILP. See `PuLP documentation <https://coin-or.github.io/pulp/technical/solvers.html>`_ for the list of supported solvers.
If not provided, ELF will use the first available solver it finds.

ELF_ILP_TIME_LIMIT
-------------------

The time limit to solve the ILP, in seconds.

ELF_ILP_GAP_REL_LIMIT
----------------------

The relative gap limit for the ILP. The first solution within this gap is returned.

ELF_CTAS_PER_COMM
------------------

Number of channels used by NCCL per communicator (i.e., per pair of ranks + direction, forward or backward)

ELF_SINGLE_P2P_COMMUNICATOR
---------------------------

When set to 1, use a single communicator (i.e. CUDA stream) for every communication. Sequentializes communications. Debug purpose.