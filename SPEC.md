# SPEC.md: "The Human How"; Exact technical requirements, environment setup, implementation details, etc.

## Execution Description

Runs torchvision ResNet-50 eval inference on synthetic images while sweeping yaml batch_size. dtype: bf16 by default (fp16 if yaml lacks bf16). image_size, warmup_iters, and num_iterations come from yaml. device_id, weights, image_source, and num_workers are ignored. Sweep dimensions: device_id, weights, dtype, precision, image_source, image_size, batch_size, num_workers.

## Parameters

| Parameter | CLI Flag | Tested Values | Default | Description |
| --- | --- | --- | --- | --- |
| device_id | `--device-id` | smoke=0, baseline=0, extended=0 | 0 | From Parameter list; see Execution Description With Parameters. |
| weights | `--weights` | smoke=random, baseline=random, extended=random | random | From Parameter list; see Execution Description With Parameters. |
| dtype | `--dtype` | smoke=bf16, baseline=bf16, extended=bf16 | bf16 | From Parameter list; see Execution Description With Parameters. |
| precision | `--precision` | smoke=bf16, baseline=bf16, extended=bf16 | bf16 | From Parameter list; see Execution Description With Parameters. |
| image_source | `--image-source` | smoke=synthetic, baseline=synthetic, extended=synthetic | synthetic | From Parameter list; see Execution Description With Parameters. |
| image_size | `--image-size` | smoke=64, baseline=224, extended=224 | 224 | From Parameter list; see Execution Description With Parameters. |
| batch_size | `--batch-size` | smoke=1,2, baseline=1,2,4,8,16,32, extended=1,2,4,8,16,32,64,128 | 1,2,4,8,16,32 | From Parameter list; see Execution Description With Parameters. |
| num_workers | `--num-workers` | smoke=0, baseline=0, extended=0 | 0 | From Parameter list; see Execution Description With Parameters. |
| warmup_iters | `--warmup-iters` | smoke=1, baseline=5, extended=10 | 5 | From Parameter list; see Execution Description With Parameters. |
| num_iterations | `--num-iterations` | smoke=3, baseline=8400, extended=13830 | 8400 | From Parameter list; see Execution Description With Parameters. |

## Invocation

```bash
Run in-process torchvision ResNet-50 eval on synthetic tensors while sampling GPU utilization during the timed loop
```

## Raw Output Format

CSV with one row per batch size. Headline metrics are copied onto every row, and case_* columns keep that batch's own values

sample_index,batch_size,batch_p99_latency_msec,per_image_latency_ms,images_sec,gpu_util_percent,peak_gpu_memory_GB,headline_batch_size
0,8,12.0,1.5,500,90,4.0,8

## Metrics

- **#1: Largest-batch inference throughput, images/s** — stored as `images_sec`.
- **#2: Largest-batch per-image latency, ms** — stored as `per_image_latency_ms`.
- **#3: Largest-batch latency max, ms** — stored as `batch_latency_max_msec`.
- **#4: Largest-batch GPU utilization** — stored as `gpu_util_percent`.
- **#5: Peak GPU memory at largest batch, GB** — stored as `peak_gpu_memory_GB`.

## Framework

Runs torchvision ResNet-50 eval inference on synthetic images while sweeping yaml batch_size. dtype: bf16 by default (fp16 if yaml lacks bf16). image_size, warmup_iters, and num_iterations come from yaml.

## Installation and Execution Summary

Run inline torchvision ResNet-50 eval over the yaml batch_size list using random weights and synthetic bf16 images, emit one row per batch size, to measure inference images/s and latency. This is not a five-minute server loop

## Platform Portability

- **AMD (primary):** ```bash
Run in-process torchvision ResNet-50 eval on synthetic tensors while sampling GPU utilization during the timed loop
```
- **NVIDIA:** Primary target is AMD ROCm. NVIDIA notes in this section are reference only and are not the execution path.

## Model Context Protocols

- **Active:** None

## Execution-Loop Validation Contract

EXECUTION CHAIN: `run_benchmark.sh` ➔ raw output ➔ `scripts/parse_results.py` ➔ `results/benchmark.db` ➔ `scripts/validate_results.py`

This benchmark uses a lightweight, SQLite-integrated execution loop for result validation. All validation is performed by `scripts/validate_results.py`.

### Validation script usage

```bash
export BENCHMARK_PYTHON=/usr/bin/python3.13  # optional; select the installed interpreter

# After a live run:
".venv/bin/python" scripts/validate_results.py --db results/benchmark.db

# CI / no-GPU path (seeds fixture and validates it):
".venv/bin/python" scripts/validate_results.py --seed-fixture --quiet

# Override DB path via environment variable:
BENCHMARK_DB=tests/fixtures/benchmark.db \
  ".venv/bin/python" scripts/validate_results.py
```

### Run artifact contract

CSV with one row per batch size. Headline metrics are copied onto every row, and case_* columns keep that batch's own values

sample_index,batch_size,batch_p99_latency_msec,per_image_latency_ms,images_sec,gpu_util_percent,peak_gpu_memory_GB,headline_batch_size
0,8,12.0,1.5,500,90,4.0,8

```bash
bash run_benchmark.sh --help
bash run_benchmark.sh --profile smoke --validate
bash run_benchmark.sh --profile baseline --validate
bash run_benchmark.sh --profile extended --validate
```
`run_benchmark.sh --help` prints usage and exits. The harness calls `scripts/ensure_setup.sh` when `.setup_state` is absent.

### Required integrity checks (built into `validate_results.py`)

1. Latest run exists and `runs.status = 'ok'`.
2. `run.error_message` is NULL.
3. `started_at` and `finished_at` are valid ISO-8601 UTC strings.
4. All required aggregate metrics in `runs` are non-NULL and finite.
5. All required aggregate metrics are physically sensible (positive values). Runs torchvision ResNet-50 eval inference on synthetic images while sweeping yaml batch_size. dtype: bf16 by default (fp16 if yaml lacks bf16). image_size, warmup_iters, and num_iterations come from yaml.
6. At least 2 sample rows exist for the latest `run_id` (sweep coverage).
7. No sample has `status = 'error'`.
8. Runs torchvision ResNet-50 eval inference on synthetic images while sweeping yaml batch_size. dtype: bf16 by default (fp16 if yaml lacks bf16). image_size, warmup_iters, and num_iterations come from yaml.

### Baseline / Threshold configuration (`config/benchmark_config.yaml`)

Expected ranges and gates live in `config/benchmark_config.yaml` under `baselines:` or `thresholds:`. To update them, edit that file — never edit validation code directly.

Threshold key suffixes encode comparison direction when `thresholds:` is present: `_min` → observed value must be ≥ threshold. `_max` → observed value must be ≤ threshold. Informational `baselines:` ranges are not pass/fail gates.
