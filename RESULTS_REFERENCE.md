# Results: 322 ResNet-50 BF16 Inference Throughput Sweep

**Main goal:** Measure ResNet-50 BF16 inference throughput  
**Tools:** Run in-process torchvision ResNet-50 eval on synthetic tensors while sampling GPU utilization during the timed loop

Reference results from the validation run on the system shown in each section (recorded 2026-11-22). Your own runs write `results/RESULTS_ACTUAL.md` in the same format.

> Smoke runs use deliberately small sizes. They check that the workload installs and runs end to end; their numbers are not performance results. Baseline and extended are.

<!-- profile:smoke -->
## smoke

- **System:** 1× AMD Instinct MI300X VF (196288 MB) · INTEL(R) XEON(R) PLATINUM 8568Y+ · Ubuntu 26.04.1 LTS · kernel 7.0.0-43-generic · driver 7.1.9.31600000
- **Run:** 2026-11-22T11:14:19Z · status **PASS** (exit 0)
- **Time (mm:ss):** setup 01:04 · benchmark 00:03 · whole run 01:17
- **Source:** run 322_20261122_111419
- **Command:** `bash run_benchmark.sh --smoke --device-id 0 --dtype bf16 --image-size 64 --batch-size 1,2 --warmup-iters 1 --num-iterations 3 --use-channels-last true`

| # | Metric | Result |
|---|---|---|
| 1 | Largest-batch inference throughput, images/s (images_sec) | 413.25659336018344 |
| 2 | Largest-batch per-image latency, ms (per_image_latency_ms) | 2.419804102504486 |
| 3 | Largest-batch latency max, ms (batch_latency_max_msec) | 7.107796000127564 |
| 4 | Largest-batch GPU utilization (gpu_util_percent) | 86.2 |
| 5 | Peak GPU memory at largest batch, GB (peak_gpu_memory_GB) | 0.144786432 |
<!-- /profile:smoke -->
