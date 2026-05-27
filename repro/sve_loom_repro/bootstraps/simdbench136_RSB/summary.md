# AST Semantic Bootstrap Summary

- total: 136
- ast_ok: 136/136

## Route Styles

- RSB_pseudocode: 136

## Patterns

- predicate_select_store_map: 58
- reduction_or_two_pass: 32
- generic_code_like_pseudocode: 13
- store_map: 11
- scan_or_prefix_monoid: 6
- bit_shift_or_rotate: 5
- scalar_or_irregular_control: 4
- two_pass_statistic: 2
- fixed_point_numeric_map: 1
- reverse_or_two_pointer_swap: 1
- numeric_conversion_lane_mapping: 1
- stencil_or_window: 1
- compact_or_streaming_write: 1

## Samples

### SimdBench_0_SVE

- route: RSB_pseudocode / predicate_select_store_map
- features: array_read, array_write, control_or_hybrid, loop_nest, predicate_or_select_map

```text
void conditional_move_simd(const int64_t *src, int64_t *dst, const bool *mask, size_t length):
  for vec_base in vector_range(0, length):
    lane i = vec_base + lane_id
    where active = i < length:
      where mask[i]:
        read: src[i]
        write: dst[i] = src[i]
```

### SimdBench_1_SVE

- route: RSB_pseudocode / fixed_point_numeric_map
- features: array_read, array_write, bitwise_or_shift, fixed_point_numeric_map, loop_nest

```text
operator_semantics:
  /: C++ division
  %: C++ remainder
  &: bitwise AND
  roundf: C++ roundf

void load_modify_store_simd(const int *src, int *dst, int scale, size_t length):
  for vec_base in vector_range(0, length):
    lane i = vec_base + lane_id
    where active = i < length:
      read: src[i]
      compute: int tmp1 = (src[i] * scale) >> 3
      read: src[i]
      compute: int tmp2 = (src[i] * scale) << 3
      read: src[i]
      write: dst[i] = (src[i] * scale + tmp1 * tmp2) / 7
```

### SimdBench_2_SVE

- route: RSB_pseudocode / store_map
- features: array_read, array_write, loop_nest, strided_or_flattened_layout

```text
void strided_load_store_simd(const double *src, double *dst, size_t rows, size_t cols, size_t stride):
  for vec_base in vector_range(0, rows):
    lane r = vec_base + lane_id
    where active = r < rows:
      read: src[r * cols + stride]
      write: dst[r] = src[r * cols + stride]
```

### SimdBench_3_SVE

- route: RSB_pseudocode / store_map
- features: array_read, array_write, loop_nest, nested_loop

```text
void indexed_access_simd(const float *src, const int *indices, float *dst, size_t length):
  for vec_base in vector_range(0, length):
    lane i = vec_base + lane_id
    where active = i < length:
      read: src[indices[i]], indices[i]
      write: dst[i] = src[indices[i]]
  for vec_base in vector_range(0, length):
    lane i = vec_base + lane_id
    where active = i < length:
      read: src[i], indices[i]
      write: dst[indices[i]] = src[i]
```

### SimdBench_4_SVE

- route: RSB_pseudocode / reverse_or_two_pointer_swap
- features: array_read, array_write, carried_dependency_or_raw, control_or_hybrid, loop_nest

```text
operator_semantics:
  /: C++ division
  %: C++ remainder
  &: bitwise AND
  roundf: C++ roundf

void range_reverse_simd(int16_t *arr, uint64_t start, uint64_t end):
  for vec_base in vector_range(0, ((end - start) + 1) / 2):
    lane offset = vec_base + lane_id
    where active = offset < ((end - start) + 1) / 2:
      lane left = start + offset
      lane right = end - offset
      where left < right:
        read: arr[left], arr[right]
        compute: temp = arr[left]
        write: arr[left] = arr[right]
        write: arr[right] = temp
```

### SimdBench_5_SVE

- route: RSB_pseudocode / store_map
- features: array_read, array_write, loop_nest, nested_loop, strided_or_flattened_layout

```text
void extract_tensor_slice_simd(const uint8_t *tensor, uint8_t *slice, size_t dim1, size_t dim2, size_t dim3, size_t slice_idx):
  for (size_t i = 0; i < dim1; i++):
    for vec_base in vector_range(0, dim2):
      lane j = vec_base + lane_id
      where active = j < dim2:
        read: tensor[(i * dim2 * dim3) + (j * dim3) + slice_idx]
        write: slice[i * dim2 + j] = tensor[(i * dim2 * dim3) + (j * dim3) + slice_idx]
```

### SimdBench_6_SVE

- route: RSB_pseudocode / store_map
- features: array_read, array_write, loop_nest, nested_loop, strided_or_flattened_layout

```text
void blocked_matrix_transpose_simd(const int16_t *src, int16_t *dst, size_t rows, size_t cols, size_t block_size):
  for (size_t i = 0; i < rows; i += block_size):
    for (size_t j = 0; j < cols; j += block_size):
      compute: std::size_t i_end = (i + block_size) < rows ? (i + block_size) : rows
      compute: std::size_t j_end = (j + block_size) < cols ? (j + block_size) : cols
      for (size_t ii = i; ii < i_end; ii++):
        for vec_base in vector_range(j, j_end):
          lane jj = vec_base + lane_id
          where active = jj < j_end:
            read: src[ii * cols + jj]
            write: dst[jj * rows + ii] = src[ii * cols + jj]
```

### SimdBench_7_SVE

- route: RSB_pseudocode / reduction_or_two_pass
- features: array_read, loop_nest, reduction_or_accumulator, scalar_reduction_return, strided_or_flattened_layout

```text
float diagonal_sum_3d_simd(const float *array, size_t dim):
  init: float sum = 0.0f
  for vec_base in vector_range(0, dim):
    lane i = vec_base + lane_id
    where active = i < dim:
      read: array[i * dim * dim + i * dim + i]
      reduce: sum += array[i * dim * dim + i * dim + i]
  return: sum
```

### SimdBench_8_SVE

- route: RSB_pseudocode / store_map
- features: array_read, array_write, loop_nest, nested_loop, strided_or_flattened_layout

```text
void conditional_scale_simd(const double *src, double *dst, size_t rows, size_t cols, double threshold, double scale):
  for vec_base in vector_range(0, rows):
    lane i = vec_base + lane_id
    where active = i < rows:
      for vec_base in vector_range(0, cols):
        lane j = vec_base + lane_id
        where active = j < cols:
          read: src[i * cols + j]
          compute: float val = src[i * cols + j]
          write: dst[i * cols + j] = (val > threshold) ? val * scale : val
```

### SimdBench_9_SVE

- route: RSB_pseudocode / store_map
- features: array_read, array_write, loop_nest, nested_loop, strided_or_flattened_layout

```text
void reorder_matrix_rows_simd(const double *src, double *dst, size_t rows, size_t cols, const size_t *indices):
  for vec_base in vector_range(0, rows):
    lane i = vec_base + lane_id
    where active = i < rows:
      read: indices[i]
      compute: std::size_t src_row = indices[i]
      for vec_base in vector_range(0, cols):
        lane j = vec_base + lane_id
        where active = j < cols:
          read: src[src_row * cols + j]
          write: dst[i * cols + j] = src[src_row * cols + j]
```

### SimdBench_10_SVE

- route: RSB_pseudocode / generic_code_like_pseudocode
- features: array_read, array_write, loop_nest

```text
operator_semantics:
  /: C++ division
  %: C++ remainder
  &: bitwise AND
  roundf: C++ roundf

void vector_vector_add_simd(const int64_t *src1, const int64_t *src2, int64_t *dst, int64_t scalar, size_t length):
  for vec_base in vector_range(0, length):
    lane i = vec_base + lane_id
    where active = i < length:
      read: src1[i], src2[i]
      write: dst[i] = (src1[i] + src2[i] + scalar) / 2
```

### SimdBench_11_SVE

- route: RSB_pseudocode / predicate_select_store_map
- features: array_read, bitwise_or_shift, control_or_hybrid, loop_nest, predicate_or_select_map

```text
operator_semantics:
  /: C++ division
  %: C++ remainder
  &: bitwise AND
  roundf: C++ roundf

int vector_even_min_simd(const int *src, size_t length):
  if src == NULL || length <= 0:
    return: -1
  read: src[0]
  compute: int min_val = src[0]
  for (size_t i = 1; i < length; i++):
    if src[i] < min_val && i % 2 == 0:
      read: src[i]
      compute: min_val = src[i]
  return: min_val
```

### SimdBench_12_SVE

- route: RSB_pseudocode / bit_shift_or_rotate
- features: array_read, array_write, bitwise_or_shift, loop_nest

```text
void mixed_right_shift_simd(const int *src, int *dst, uint8_t shift, size_t length):
  for vec_base in vector_range(0, length):
    lane i = vec_base + lane_id
    where active = i < length:
      read: src[i]
      write: dst[i] = src[i] >> shift
      read: src[i], dst[i]
      lane_update: dst[i] += (int)((unsigned int)src[i] >> shift)
```

### SimdBench_13_SVE

- route: RSB_pseudocode / reduction_or_two_pass
- features: array_read, loop_nest, nested_loop, reduction_or_accumulator, scalar_reduction_return, strided_or_flattened_layout

```text
int64_t matrix_sum_simd(const int *matrix, size_t rows, size_t cols):
  init: int64_t sum = 0
  for vec_base in vector_range(0, rows):
    lane i = vec_base + lane_id
    where active = i < rows:
      scalar_region:
        for (size_t j = 0; j < cols; j++):
          read: matrix[i * cols + j]
          reduce: sum += matrix[i * cols + j]
  return: sum
```

### SimdBench_14_SVE

- route: RSB_pseudocode / predicate_select_store_map
- features: control_or_hybrid, loop_nest, predicate_or_select_map

```text
size_t argmax_simd(const int8_t *src, size_t length):
  if src == NULL:
    return: size_t(0)
  init: std::size_t index = 0
  for (size_t i = 1; i < length; i++):
    if src[i] > src[index]:
      update: index = i
  return: index
```

### SimdBench_15_SVE

- route: RSB_pseudocode / bit_shift_or_rotate
- features: array_read, array_write, bitwise_or_shift, loop_nest

```text
operator_semantics:
  /: C++ division
  %: C++ remainder
  &: bitwise AND
  roundf: C++ roundf

void vector_blend_simd(const uint16_t* src1, const uint16_t* src2, uint32_t mask, uint16_t* dst, size_t length):
  for vec_base in vector_range(0, length):
    lane i = vec_base + lane_id
    where active = i < length:
      read: src1[i], src2[i]
      write: dst[i] = (i & mask) ? src1[i] : src2[i]
```

### SimdBench_16_SVE

- route: RSB_pseudocode / bit_shift_or_rotate
- features: array_read, array_write, bitwise_or_shift, loop_nest

```text
operator_semantics:
  /: C++ division
  %: C++ remainder
  &: bitwise AND
  roundf: C++ roundf

void population_count_simd(const uint32_t* src, uint32_t* dst, size_t length):
  for vec_base in vector_range(0, length):
    lane i = vec_base + lane_id
    where active = i < length:
      read: src[i]
      compute: uint32_t x = src[i]
      state_reduce: x = x - ((x >> 1) & 0x55555555)
      state_reduce: x = (x & 0x33333333) + ((x >> 2) & 0x33333333)
      write: dst[i] = (((x + (x >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24
```

### SimdBench_17_SVE

- route: RSB_pseudocode / generic_code_like_pseudocode
- features: array_read, array_write, loop_nest

```text
void saturating_add_simd(const uint16_t* src1, const uint16_t* src2, uint16_t* dst, size_t length):
  for vec_base in vector_range(0, length):
    lane i = vec_base + lane_id
    where active = i < length:
      read: src1[i], src2[i]
      compute: uint16_t sum = src1[i] + src2[i]
      read: src1[i]
      write: dst[i] = (sum < src1[i]) ? UINT16_MAX : sum
```

### SimdBench_18_SVE

- route: RSB_pseudocode / reduction_or_two_pass
- features: array_read, array_write, bitwise_or_shift, loop_nest, nested_loop, predicate_or_select_map, reduction_or_accumulator

```text
void range_matrix_mul_simd(const double* A, const double* B, double* C, size_t m, size_t n, size_t p):
  for (size_t i = 0; i < m; i++):
    for (size_t j = 0; j < p; j++):
      init: double sum = 0.0
      for (size_t k = 0; k < n; k++):
        read: A[i * n + k]
        compute: double a_val = A[i * n + k]
        read: B[k * p + j]
        compute: double b_val = B[k * p + j]
        if a_val >= -100 && a_val <= 100 && b_val >= -100 && b_val <= 100:
          update: sum += a_val * b_val
      write: C[i*p + j] = sum
```

### SimdBench_19_SVE

- route: RSB_pseudocode / store_map
- features: array_read, array_write, loop_nest, nested_loop

```text
void tensor_add_3d_simd(const int64_t* A, const int64_t* B, int64_t* C, size_t dim1, size_t dim2, size_t dim3):
  for vec_base in vector_range(0, dim1):
    lane i = vec_base + lane_id
    where active = i < dim1:
      for vec_base in vector_range(0, dim2):
        lane j = vec_base + lane_id
        where active = j < dim2:
          for vec_base in vector_range(0, dim3):
            lane k = vec_base + lane_id
            where active = k < dim3:
              compute: std::size_t idx = i*dim2*dim3 + j*dim3 + k
              read: A[idx], B[idx]
              write: C[idx] = A[idx] + B[idx]
```
