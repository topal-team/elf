# Test-suite layout

This repository now separates **fast unit checks** from **GPU / multi-process integration**.
Everything is driven by *pytest* markers so you can cherry-pick exactly what you want to run.

```
 tests/
 ├─ unit/              # ≤ a few seconds, CPU-only by default
 │   ├─ utils/         # elf.utils helpers
 │   ├─ partitioners/  # tracing / partition logic
 │   └─ scheduling/    # static schedule generation
 │
 ├─ distributed/       # require CUDA devices + torch.distributed
 │   ├─ execution/     # PipelineBlock P2P logic
 │   └─ pipeline/      # end-to-end pipeline runs
 │
 └─ conftest.py        # auto-skip GPU / distributed tests when env is missing
```

## Markers
The list is declared in `pytest.ini`:

| marker          | meaning | typical runtime |
|-----------------|---------|-----------------|
| `unit`          | CPU-only, quick           | < 1 s |
| `integration`   | single-process GPU tests  | ~ few s |
| `distributed`   | multi-process / NCCL      | depends on model |
| `gpu`           | needs at least one CUDA device | |
| `slow`          | opt-in checks that take noticeable time | |
| `single` / `multi` | legacy aliases kept for backward compatibility | |

> `conftest.py` automatically skips `gpu` / `distributed` tests if the environment cannot satisfy their requirements, so `pytest` is always safe to run on any CI worker.

## Quick recipes
Activate your venv first (user rule):
```bash
source /home/adrien/venv/bin/activate
```

Run everything that is guaranteed to be fast and CPU-safe:
```bash
pytest -m unit -q
```

Run **all** tests that don’t need multiple ranks (CPU *and* single-GPU):
```bash
pytest -m "not distributed" -q
```

Full distributed suite on 4 GPUs:
```bash
torchrun --standalone --nproc_per_node 4 \
        -m pytest -m distributed --runslow -q
```

## Legacy helpers
`tests/test.sh` still exists for backwards compatibility but simply calls the new marker layout; feel free to remove once your CI is updated.

## What each folder covers
* **unit/utils** – Timer, Placement, NameMapping, TensorMetadata, etc.
* **unit/partitioners** – FX tracing, inplace-op removal, Metis/Custom partition correctness, operation profiling.
* **unit/scheduling** – golden patterns for AFAB / 1F1B schedulers and helper invariants.
* **distributed/execution** – single-rank logic plus 2-GPU Variable communication path.
* **distributed/pipeline** – 4-GPU pipeline creation, correctness, inference path.

This structure lets you iterate quickly while still guarding the heavyweight distributed code-paths before each release.