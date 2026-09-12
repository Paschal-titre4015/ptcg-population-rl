# Measure C++ inference and self-play performance

Pass a trained, exported Mega Lucario model with `--model`. Replace the example model paths with your own outputs. Run commands from the repository root.

## Inference latency

```bash
uv run python src/agents/mega_lucario/transformer/scripts/benchmark.py \
  --model models/mega_lucario/transformer/model.bin \
  --output-dir outputs/benchmark_forward \
  --repetitions 100 --tokens 32 50 112 120
```

Each input receives three warmup runs before timing the requested number of repetitions. The measurement includes input conversion, Encoder, Policy, and Value; it excludes model loading, game progression, and feature extraction from observations.
Available scalar, AVX2, and OpenBLAS CPU paths are compared with a double-precision reference. Timings and environment details are saved to `benchmark.json`. OpenBLAS uses one thread.

## Self-play throughput

```bash
uv run python src/agents/mega_lucario/transformer/scripts/benchmark_selfplay.py \
  --before build/arena_cpp \
  --model models/mega_lucario/transformer/model.bin \
  --output-dir outputs/benchmark_selfplay \
  --games 16 --repetitions 3 --replay --sampled
```

The measurement uses one process and one OpenBLAS thread. It includes process startup, model loading, feature extraction, inference, game progression, and requested recording. Use an output directory that does not exist yet.
Omit `--replay --sampled` for deterministic actions without recording.

To compare changes, pass the saved original executable with `--before` and the new executable with `--after`.
The benchmark alternates execution order, uses identical models and seeds, and compares results and decision counts. With recording enabled, it also compares action sequences. Results are saved to `benchmark.json`.
