# High-Quality ARM SIMD Loops Serial References

Generated on .

## Inputs

- Original ARM SIMD Loops problem file:
  `/home/user/selective_repo/results/generated/contextualized_benchmarks/arm_simd_loops/arm_simd_loops_contextualized.problem.with_solution_scalar.jsonl`

## Outputs

- New serial-code artifact:
  `/home/user/simdbench_full/results/generated/high_quality_arm_simd_loops/arm_simd_loops54.new_serial_code.v1.jsonl`
- Problem overlay using the new serial references:
  `/home/user/simdbench_full/results/generated/high_quality_arm_simd_loops/arm_simd_loops54.problem.high_quality_serial_v1.jsonl`
- Local correctness report:
  `/home/user/simdbench_full/results/generated/high_quality_arm_simd_loops/arm_simd_loops54.new_serial_code.v1.correctness_report.json`
- Latest RSB bootstrap directory pointer:
  `/home/user/simdbench_full/results/generated/high_quality_arm_simd_loops/latest_bootstrap_dir.txt`

## Validation Summary

- Rows: 54
- Manual high-quality overrides: 32
- Original serial passthrough rows: 22
- Local correctness smoke: 53/54 passed
- Local skip: `arm_simd_loops.loop_038`, because local x86 `g++` does not support `__fp16` function signatures.
- RSB bootstrap: 54/54 AST parsed and 54/54 non-empty.

## Performance Note

The current ARM SIMD Loops problem file does not contain a uniform performance harness or `test_performance` field. This version is therefore a correctness-checked, speed-oriented serial-reference candidate set, not a verified all-speedup-greater-than-1 set. A strict speedup gate requires adding an ARM SIMD Loops serial-vs-original performance oracle.
