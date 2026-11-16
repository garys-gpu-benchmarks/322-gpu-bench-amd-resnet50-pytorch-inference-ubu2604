# Results: 322 ResNet-50 BF16 Inference Throughput Sweep

**Main goal:** Measure ResNet-50 BF16 inference throughput  
**Tools:** Run in-process torchvision ResNet-50 eval on synthetic tensors while sampling GPU utilization during the timed loop

Reference results measured by the maintainer on the system shown in each section (generator TEMPLATE_00_143 · recorded 2026-11-16). Your own runs write `results/RESULTS_ACTUAL.md` in the same format.

> Smoke runs use deliberately small sizes. They check that the workload installs and runs end to end; their numbers are not performance results. Baseline and extended are.

<!-- profile:smoke -->
## smoke

- **System:** 1× AMD Instinct MI300X VF (196288 MB) · INTEL(R) XEON(R) PLATINUM 8568Y+ · Ubuntu 26.04.1 LTS · kernel 7.0.0-38-generic · driver 7.1.9.31600000
- **Run:** 2026-11-16T19:54:23Z · status **PASS** (exit 0)
- **Time (mm:ss):** setup 01:08 · benchmark 00:04 · whole run 01:21
- **Source:** run 322_20261116_195423
- **Command:** `bash run_benchmark.sh --smoke --device-id 0 --dtype bf16 --image-size 64 --batch-size 1,2 --warmup-iters 1 --num-iterations 3 --use-channels-last true`

| # | Metric | Result |
|---|---|---|
| 1 | Largest-batch inference throughput, images/s (images_sec) | 451.69003593053657 |
| 2 | Largest-batch per-image latency, ms (per_image_latency_ms) | 2.2139075924928875 |
| 3 | Largest-batch latency max, ms (batch_latency_max_msec) | 4.928300000301533 |
| 4 | Largest-batch GPU utilization (gpu_util_percent) | 84.66666666666667 |
| 5 | Peak GPU memory at largest batch, GB (peak_gpu_memory_GB) | 0.144786432 |
<!-- /profile:smoke -->
