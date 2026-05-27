# SVE-Loom Artifact

This repository contains the anonymous reproducibility artifact for a system that
generates Arm SVE code from natural-language task descriptions.  The artifact is
organized to support three parts of the paper:

1. the post-filtering NL-to-SVE training corpus,
2. the SIMDBench and ARM SIMD Loops evaluation setup, and
3. the SVE-Loom bootstrap, inference, feedback, and ACLE whitelist tooling.

The repository intentionally contains only the files needed for reproduction.
Model checkpoints, adapter weights, raw nohup logs, intermediate failed
candidates, and old backup snapshots are not included.

## Repository Layout

```text
.
├── data/whitelists/
│   └── Clang 15 SVE/ACLE whitelist artifacts
├── repro/sve_loom_repro/
│   ├── data/
│   │   ├── train.post_highrisk_holdout.jsonl
│   │   ├── dev.post_highrisk_holdout.jsonl
│   │   └── decontamination and high-risk holdout audit files
│   ├── benchmarks/
│   │   ├── simdbench/
│   │   └── arm_simd_loops/
│   ├── bootstraps/
│   │   ├── simdbench136_RSB/
│   │   ├── simdbench136_high_quality_v20/
│   │   ├── arm_simd_loops54_RSB/
│   │   └── arm_simd_loops54_high_quality_serial_v1/
│   ├── scripts/
│   ├── simdbench_pkg/
│   └── whitelists/
└── scripts/
    ├── build_ast_semantic_bootstrap.py
    ├── build_contextualized_benchmark_problem_files.py
    ├── build_sve_acle_whitelist.py
    ├── audit_*duplicate*.py
    └── inference/
```

`repro/sve_loom_repro/FILES.txt` lists every file included in the
reproducibility bundle.

## Corpus Snapshot

The post-high-risk-holdout corpus snapshot contains:

| Split | File | Rows |
| --- | --- | ---: |
| Train | `repro/sve_loom_repro/data/train.post_highrisk_holdout.jsonl` | 5,321 |
| Dev | `repro/sve_loom_repro/data/dev.post_highrisk_holdout.jsonl` | 626 |

The audit and isolation files are:

- `repro/sve_loom_repro/data/nonoverlap_audit_summary.json`
- `repro/sve_loom_repro/data/nonoverlap_audit_summary.md`
- `repro/sve_loom_repro/data/nonoverlap_highrisk29_holdout_manifest.tsv`

These files document the duplicate and high-risk held-out overlap checks used
before training.

## Evaluation Files

SIMDBench files are under:

```text
repro/sve_loom_repro/benchmarks/simdbench/
```

Important files:

- `simdbench_sve.jsonl`: main SIMDBench SVE task file.
- `simdbench_sve_hidden_strict.jsonl`: strict hidden-test task file.
- `simdbench136.v20_mixed_original6.problem_overlay.jsonl`:
  high-quality serial overlay used for the v20 RSB bootstrap.
- `simdbench136.new_serial_code.v20_mixed_original6.jsonl`: generated
  high-quality serial reference candidates.

ARM SIMD Loops files are under:

```text
repro/sve_loom_repro/benchmarks/arm_simd_loops/
```

Important files:

- `arm_simd_loops_contextualized.problem.with_solution_scalar.jsonl`:
  main ARM SIMD Loops task file.
- `arm_simd_loops_contextualized.remote_eval.cxx17_hidden_strict.problem.jsonl`:
  strict hidden-test problem file.
- `arm_simd_loops54.problem.high_quality_serial_v1.jsonl`: high-quality serial
  overlay for ARM SIMD Loops.
- `arm_simd_loops54.new_serial_code.v1.jsonl`: generated high-quality serial
  reference candidates.

## Bootstrap Files

The RSB semantic bootstrap files used in the paper are:

```text
repro/sve_loom_repro/bootstraps/simdbench136_RSB/ast_semantic_bootstrap.jsonl
repro/sve_loom_repro/bootstraps/simdbench136_high_quality_v20/ast_semantic_bootstrap.jsonl
repro/sve_loom_repro/bootstraps/arm_simd_loops54_RSB/ast_semantic_bootstrap.jsonl
repro/sve_loom_repro/bootstraps/arm_simd_loops54_high_quality_serial_v1/ast_semantic_bootstrap.jsonl
```

Each bootstrap directory also includes a `summary.md` file with task-level
rendering summaries.

The canonical builder is:

```text
scripts/build_ast_semantic_bootstrap.py
```

## Training Entry Points

Training scripts included in the bundle:

```text
repro/sve_loom_repro/scripts/train_sft_new_qlora.py
repro/sve_loom_repro/scripts/train_v22_direct_codegen_lora32_drop01_epoch3_ns5_t08_full.sh
repro/sve_loom_repro/scripts/train_v22_direct_codegen_legacy1280_lora_env.sh
```

The shell scripts expose the model path, data directory, and output directory as
environment variables.  Set these variables to local paths before launching a
training run.

## Inference and Feedback Entry Points

The canonical inference script is:

```text
scripts/inference/gen_simdbench_with_repaires_0128_seq_feedback_t02_r01_earlystop_pass1_v22_phased_serial_bootstrap.py
```

The structured compile-feedback audit helper is:

```text
scripts/inference/audit_structured_compile_feedback_v22.py
```

The hidden-test construction and evaluation helpers are:

```text
repro/sve_loom_repro/scripts/build_hidden_strict_eval_problem_files.py
repro/sve_loom_repro/scripts/correctness_eval.py
repro/sve_loom_repro/scripts/remote_eval_simdbench_perf.py
```

## ACLE Whitelist

The SVE/ACLE whitelist used by Name/ShapeFix is included in two locations:

```text
data/whitelists/
repro/sve_loom_repro/whitelists/
```

The whitelist builder is:

```text
scripts/build_sve_acle_whitelist.py
```

The primary whitelist artifact is:

```text
whitelist.sve.clang15.json
```

## Environment Notes

The artifact assumes:

- Python 3.10 or newer for the provided scripts.
- A Clang/LLVM toolchain with Arm SVE ACLE headers for compile validation.
- An AArch64/SVE-capable target or remote evaluator for execution and
  performance measurement.
- PyTorch, Transformers, PEFT, and BitsAndBytes for QLoRA/SFT training.

API keys, private SSH keys, model checkpoints, and adapter weights are not
included.  Scripts that call remote models or remote evaluators expect credentials
to be supplied through environment variables or command-line arguments.

## Quick Checks

Count the corpus rows:

```bash
wc -l repro/sve_loom_repro/data/train.post_highrisk_holdout.jsonl
wc -l repro/sve_loom_repro/data/dev.post_highrisk_holdout.jsonl
```

List the included files:

```bash
sed -n '1,120p' repro/sve_loom_repro/FILES.txt
```

Inspect a bootstrap summary:

```bash
sed -n '1,120p' repro/sve_loom_repro/bootstraps/simdbench136_high_quality_v20/summary.md
```

## Anonymity

This artifact is prepared for anonymous review.  The repository history contains
a single anonymous commit, and the files are organized without author-identifying
repository metadata.
