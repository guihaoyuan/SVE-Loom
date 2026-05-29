# SVE-Loom Reproducibility Bundle

This directory contains the minimal files needed to reproduce the paper-facing
SVE-Loom training/evaluation setup without depending on the original local
workspace.

## Included

- `data/`
  - `train.curated.jsonl`: curated training split, 5,321 rows.
  - `dev.curated.jsonl`: curated development split, 626 rows.
  - holdout/decontamination manifest and summary files.
- `benchmarks/simdbench/`
  - SIMDBench task data and strict hidden-test data.
  - High-quality serial reference overlay used for the v20 RSB bootstrap.
- `benchmarks/arm_simd_loops/`
  - ARM SIMD Loops task files, strict hidden-test problem file, and high-quality serial overlay.
- `bootstraps/`
  - SIMDBench RSB bootstrap.
  - SIMDBench high-quality serial v20 RSB bootstrap.
  - ARM SIMD Loops RSB bootstrap.
  - ARM SIMD Loops high-quality serial RSB bootstrap.
- `simdbench_pkg/`
  - Minimal SIMDBench Python package files needed by local evaluation scripts.
- `scripts/`
  - Training, inference, hidden-test construction, high-quality scalar generation,
    correctness/performance evaluation, bootstrap construction, and whitelist construction scripts.
- `whitelists/`
  - Clang 15 SVE/ACLE whitelist artifacts.

`FILES.txt` lists every file in this bundle.

## Canonical Scripts In Repository

The current canonical RSB bootstrap builder is also tracked at:

- `scripts/build_ast_semantic_bootstrap.py`

The current canonical inference script is also tracked at:

- `scripts/inference/gen_simdbench_with_repaires_0128_seq_feedback_t02_r01_earlystop_pass1_v22_phased_serial_bootstrap.py`

The whitelist builder and whitelist data are also tracked at:

- `scripts/build_sve_acle_whitelist.py`
- `data/whitelists/whitelist.sve.clang15.json`
- `data/whitelists/whitelist.sve.clang15.removed_v2.txt`

## Notes

- Model checkpoints and adapter weights are intentionally not included.
- Large raw logs, nohup logs, intermediate failed candidates, and old backup snapshots are intentionally not included.
- The ARM SIMD Loops high-quality serial overlay is correctness-checked locally for 53/54 rows; `loop_038` requires an AArch64/SVE toolchain because the local x86 compiler does not parse `__fp16` function signatures.
