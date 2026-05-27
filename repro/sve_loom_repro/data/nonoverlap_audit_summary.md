# Full Non-overlap Audit

- Core rows: 5947 (5321 train, 626 dev)
- Benchmark rows: 190 {'simdbench_sve': 136, 'arm_simd_loops': 54}

## Text / Token Audit
```json
{
  "prompt": {
    "prompt_exact": 0,
    "prompt_ngram": 0,
    "prompt_minhash": 0,
    "prompt_simhash": 0,
    "prompt_edit": 0
  },
  "scalar_code": {
    "scalar_code_exact": 0,
    "scalar_code_ngram": 0,
    "scalar_code_minhash": 0,
    "scalar_code_simhash": 0,
    "scalar_code_edit": 0
  },
  "response_code": {
    "response_code_exact": 0,
    "response_code_ngram": 0,
    "response_code_minhash": 0,
    "response_code_simhash": 0,
    "response_code_edit": 0
  }
}
```

## AST Structural Audit
```json
{
  "typed_ast_exact": 0,
  "untyped_ast_exact": 0,
  "loop_nest_exact": 35838,
  "array_access_exact": 33794,
  "assignment_tree_exact": 17,
  "expression_tree_exact": 30,
  "call_graph_exact": 150,
  "composite_ast_exact": 0
}
```

## Semantic Pattern Audit
```json
{
  "semantic_strict_exact": 0,
  "semantic_abstract_exact": 6,
  "semantic_pattern_high_risk": 0
}
```

## Identifier / Literal Audit
```json
{
  "identifier_or_literal_matches": 508,
  "non_generic_or_same_signature": 20
}
```
