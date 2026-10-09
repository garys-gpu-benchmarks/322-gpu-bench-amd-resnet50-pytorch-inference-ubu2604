# ResNet-50 BF16 Inference Throughput Sweep Benchmark

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![CI](https://github.com/garys-gpu-benchmarks/322-gpu-bench-amd-resnet50-pytorch-inference-ubu2604/actions/workflows/ci.yml/badge.svg)](https://github.com/garys-gpu-benchmarks/322-gpu-bench-amd-resnet50-pytorch-inference-ubu2604/actions/workflows/ci.yml)

Target: Ubuntu 26.04 · AMD · see Hardware Requirements. This is a host benchmark, not a laptop `pip install` project.

## Quick Start

```bash
git clone https://github.com/garys-gpu-benchmarks/322-gpu-bench-amd-resnet50-pytorch-inference-ubu2604.git
cd 322-gpu-bench-amd-resnet50-pytorch-inference-ubu2604
sudo bash setup.sh --assume-yes
bash run_benchmark.sh --profile smoke --validate
```
Results are written to `results/benchmark.db` and `results/summary.json`.

This workload is executed on the validation host after the repository is copied there. `setup.sh` and `run_benchmark.sh` do not open an outbound SSH session.

Prerequisites: Ubuntu 26.04; AMD; Python 3.14.4; root or sudo for `setup.sh`. Framework: Bash, SQLite, Python, PyYAML, ROCm Runtime, PyTorch-ROCm, torchvision, HIP/ROCm. This is a host benchmark, not a laptop `pip install` project.

```mermaid
flowchart LR
  setup.sh --> run_benchmark.sh --> parse_results.py --> results/benchmark.db
```

## 1. Overview

Runs torchvision ResNet-50 eval inference on synthetic images while sweeping yaml batch_size. dtype: bf16 by default (fp16 if yaml lacks bf16). image_size, warmup_iters, and num_iterations come from yaml. device_id, weights, image_source, and num_workers are ignored. Sweep dimensions: device_id, weights, dtype, precision, image_source, image_size, batch_size, num_workers.

## 2. What It Validates

- Validates one row per batch size from inline ResNet-50 eval. This is not a five-minute server loop
- #1: Largest-batch inference throughput, images/s (images_sec); is present and physically sensible.
- #2: Largest-batch per-image latency, ms (per_image_latency_ms); is present and physically sensible.
- #3: Largest-batch p99 latency, ms (batch_p99_latency_msec); is present and physically sensible.
- #4: Largest-batch GPU utilization (gpu_util_percent); is present and physically sensible.
- #5: Peak GPU memory at largest batch, GB (peak_gpu_memory_GB) is present and physically sensible.

## 3. Metrics Captured

- **#1: Largest-batch inference throughput, images/s** — stored as `images_sec`.
- **#2: Largest-batch per-image latency, ms** — stored as `per_image_latency_ms`.
- **#3: Largest-batch p99 latency, ms** — stored as `batch_p99_latency_msec`.
- **#4: Largest-batch GPU utilization** — stored as `gpu_util_percent`.
- **#5: Peak GPU memory at largest batch, GB** — stored as `peak_gpu_memory_GB`.

## 4. Hardware Requirements

### Supported environment

- OS: Ubuntu 26.04
- GPU vendor: AMD
- Framework family: Bash, SQLite, Python, PyYAML, ROCm Runtime, PyTorch-ROCm, torchvision, HIP/ROCm
- Python: Python 3.14.4

### Reference validation environment

The tables below describe the machine used to generate the reference results. They are not a requirement that every user buy that exact cloud instance.

### System

Ubuntu 26.04 / AMD / Bash, SQLite, Python, PyYAML, ROCm Runtime, PyTorch-ROCm, torchvision, HIP/ROCm

### GPU

Ubuntu 26.04 / AMD / Bash, SQLite, Python, PyYAML, ROCm Runtime, PyTorch-ROCm, torchvision, HIP/ROCm

## 5. Software Requirements

| Component | Version |
|---|---|
| OS | Ubuntu 26.04 |
| Kernel | kernel 7.0.0 |
| Python | Python 3.14.4 |
| ROCm | ROCm 7.14 |
| rocBLAS | N/A - rocBLAS not used |

Runs torchvision ResNet-50 eval inference on synthetic images while sweeping yaml batch_size. dtype: bf16 by default (fp16 if yaml lacks bf16). image_size, warmup_iters, and num_iterations come from yaml.

## 6. Installation

```bash
Run in-process torchvision ResNet-50 eval on synthetic tensors while sampling GPU utilization during the timed loop
```

## 7. Running the Benchmark

```bash
Run in-process torchvision ResNet-50 eval on synthetic tensors while sampling GPU utilization during the timed loop
```

**Validating results separately:**

```bash
export BENCHMARK_PYTHON=/usr/bin/python3.13  # optional
python3 -m venv .venv
source ".venv/bin/activate"
".venv/bin/python" scripts/validate_results.py
```

## 8. Output

### `results/benchmark.db` (SQLite)

CSV with one row per batch size. Headline metrics are copied onto every row, and case_* columns keep that batch's own values

sample_index,batch_size,batch_p99_latency_msec,per_image_latency_ms,images_sec,gpu_util_percent,peak_gpu_memory_GB,headline_batch_size
0,8,12.0,1.5,500,90,4.0,8

```bash
Run in-process torchvision ResNet-50 eval on synthetic tensors while sampling GPU utilization during the timed loop
```

### `results/summary.json`

Consolidated metrics from the most recent run — suitable for CI artifact upload or dashboard ingestion.

### `results/raw/<timestamp>.txt`

CSV with one row per batch size. Headline metrics are copied onto every row, and case_* columns keep that batch's own values

sample_index,batch_size,batch_p99_latency_msec,per_image_latency_ms,images_sec,gpu_util_percent,peak_gpu_memory_GB,headline_batch_size
0,8,12.0,1.5,500,90,4.0,8

## 9. Baselines / Thresholds

Expected ranges and gates live in `config/benchmark_config.yaml` under `baselines:` or `thresholds:`. To update them, edit that file — never edit validation code directly.

## 10. Troubleshooting

**`setup.sh` missing collector**
Create cannot finish without `scripts/collect_workload.py`.

**`self_check` overlay rewritten**
Do not overwrite files listed in `results/overlay_lock.json`.

**Remote SSH drop during setup**
Reconnect and resume `bash setup.sh --assume-yes`. Do not wipe `.venv` or `.cache`.

## 11. NVIDIA H100 Coding Differences

Primary target is AMD ROCm. NVIDIA notes in this section are reference only and are not the execution path.

## Repository layout

```text
.
├── setup.sh
├── run_benchmark.sh
├── benchmark_specification.json
├── .github/workflows/      # thin CI callers (see Continuous Integration)
├── config/
├── scripts/
├── src/
├── tests/
├── docs/
├── results/
└── LICENSE
```

## Continuous Integration

| Workflow | Runs on | When | What it does |
|---|---|---|---|
| [CI](.github/workflows/ci.yml) | GitHub-hosted runner | every pull request, and every push to `main` | shellcheck, ruff, `bash -n`, `compileall`, `run_benchmark.sh --help`, specification schema, the results validator on a seeded fixture, required files, and actionlint. No GPU and no benchmark run. |
| [GPU Smoke Benchmark](.github/workflows/gpu-smoke.yml) | self-hosted runner labeled `gpu`, `amd`, `ubu2604` | only when started by hand: **Actions → GPU Smoke Benchmark → Run workflow** (choose `smoke`, `baseline` or `extended`) | Verifies the pre-provisioned GPU stack, records `results/environment.json` (driver, runtime, kernel, GPU), runs the profile with `--validate`, shows headline metrics on the run page, and uploads the results. |

Both files are short callers. The steps themselves live once, for every workload in the suite, in [`garys-gpu-benchmarks/shared-workflows`](https://github.com/garys-gpu-benchmarks/shared-workflows), pinned at `@v1`. The GPU workflow is never triggered by pull requests, so code from a fork cannot run on the GPU host.

### Running it as part of the AMD Ubuntu 26.04 bundle

This repository is one of the 32 workloads in [`bundle-amd-ubuntu-2604`](https://github.com/garys-gpu-benchmarks/bundle-amd-ubuntu-2604), which holds them as git submodules. To put the whole bundle on a GPU host and run this workload from it:

```bash
git clone --recurse-submodules https://github.com/garys-gpu-benchmarks/bundle-amd-ubuntu-2604 /opt/benchmarks
cd /opt/benchmarks/322-gpu-bench-amd-resnet50-pytorch-inference-ubu2604
bash run_benchmark.sh --profile smoke --validate
```
