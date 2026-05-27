# AST Semantic Bootstrap Summary

- total: 54
- ast_ok: 54/54

## Route Styles

- RSB_pseudocode: 54

## Patterns

- reduction_or_two_pass: 19
- generic_code_like_pseudocode: 15
- scan_or_prefix_monoid: 7
- store_map: 5
- bit_shift_or_rotate: 4
- predicate_select_store_map: 3
- fixed_point_numeric_map: 1

## Samples

### arm_simd_loops.loop_001

- route: RSB_pseudocode / reduction_or_two_pass
- features: array_read, loop_nest, reduction_or_accumulator

```text
float arm_simd_loop_001(float * a, float * b, int n):
  init: float res = 0.0f
  for (size_t base = 0; base < n; base += VL<float>):
    pg = active_lanes<float>(base, n)
    lane int i = base + lane_id where pg
    read: a[i], b[i]
    reduce: res += a[i] * b[i]
  return: res
```

### arm_simd_loops.loop_002

- route: RSB_pseudocode / reduction_or_two_pass
- features: array_read, loop_nest, reduction_or_accumulator

```text
uint32_t arm_simd_loop_002(uint32_t * a, uint32_t * b, int n):
  init: uint32_t res = 0
  for (size_t base = 0; base < n; base += VL<uint32_t>):
    pg = active_lanes<uint32_t>(base, n)
    lane int i = base + lane_id where pg
    read: a[i], b[i]
    reduce: res += a[i] * b[i]
  return: res
```

### arm_simd_loops.loop_003

- route: RSB_pseudocode / reduction_or_two_pass
- features: array_read, loop_nest, reduction_or_accumulator

```text
double arm_simd_loop_003(double * a, double * b, int n):
  init: double res = 0.0
  for (size_t base = 0; base < n; base += VL<double>):
    pg = active_lanes<double>(base, n)
    lane int i = base + lane_id where pg
    read: a[i], b[i]
    reduce: res += a[i] * b[i]
  return: res
```

### arm_simd_loops.loop_004

- route: RSB_pseudocode / reduction_or_two_pass
- features: array_read, loop_nest, reduction_or_accumulator

```text
uint64_t arm_simd_loop_004(uint64_t * a, uint64_t * b, int n):
  init: uint64_t res = 0
  for (size_t base = 0; base < n; base += VL<uint64_t>):
    pg = active_lanes<uint64_t>(base, n)
    lane int i = base + lane_id where pg
    read: a[i], b[i]
    reduce: res += a[i] * b[i]
  return: res
```

### arm_simd_loops.loop_005

- route: RSB_pseudocode / reduction_or_two_pass
- features: bitwise_or_shift, loop_nest, nested_loop, reduction_or_accumulator

```text
uint32_t arm_simd_loop_005(uint8_t * p, uint8_t * lmt):
  init: uint32_t res = 0
  while p < lmt:
    init: uint32_t len = 0
    while p + len < lmt and p[len] != 0:
      update: len += 1
    update: p += len + 1
    update: res += 1
    update: res ^= (len % 0xffff) << 16
  return: res
```

### arm_simd_loops.loop_006

- route: RSB_pseudocode / reduction_or_two_pass
- features: bitwise_or_shift, loop_nest, nested_loop, reduction_or_accumulator

```text
uint32_t arm_simd_loop_006(uint8_t * p, uint8_t * lmt):
  init: uint32_t res = 0
  while p < lmt:
    init: uint32_t len = 0
    while p + len < lmt and p[len] != 0:
      update: len += 1
    update: p += len + 1
    update: res += 1
    update: res ^= (len % 0xffff) << 16
  return: res
```

### arm_simd_loops.loop_008

- route: RSB_pseudocode / reduction_or_two_pass
- features: array_read, loop_nest, reduction_or_accumulator

```text
double arm_simd_loop_008(double * a, int n):
  init: double res = 0.0
  for (size_t base = 0; base < n; base += VL<double>):
    pg = active_lanes<double>(base, n)
    lane int i = base + lane_id where pg
    read: a[i]
    reduce: res += a[i]
  return: res
```

### arm_simd_loops.loop_009

- route: RSB_pseudocode / reduction_or_two_pass
- features: bitwise_or_shift, loop_nest, reduction_or_accumulator

```text
uint64_t arm_simd_loop_009(node_t * nodes):
  init: uint64_t res = 0
  for (node_t *p = nodes; p != NULL; p = p->next):
    update: res ^= p->payload ^ p->payload2
  return: res
```

### arm_simd_loops.loop_010

- route: RSB_pseudocode / predicate_select_store_map
- features: control_or_hybrid, loop_nest, predicate_or_select_map

```text
int arm_simd_loop_010(float * a, uint64_t n):
  init: bool any = 0
  init: bool all = 1
  for (int i = 0; i < n; i++):
    if a[i] < 0.0f:
      update: any = 1
    else:
      update: all = 0
  return: all ? 1 : any ? 2 : 3
```

### arm_simd_loops.loop_012

- route: RSB_pseudocode / generic_code_like_pseudocode
- features: array_read, array_write, loop_nest

```text
void arm_simd_loop_012(int64_t step, double * direction, int64_t * magnitude, double * vx, double * vy, double * vz, double * nx, double * ny, double * nz, uint64_t n):
  for (size_t base = 0; base < n; base += VL<double>):
    pg = active_lanes<double>(base, n)
    lane int p = base + lane_id where pg
    read: direction[0]
    compute: double pos = direction[0]
    read: vx[p]
    compute: double value = vx[p]
    compute: double vabs = value < 0 ? -value : value
    read: magnitude[0]
    compute: double vabsstep = (vabs * step) - ((int64_t)(vabs * step) / magnitude[0]) * magnitude[0]
    compute: double vstep = value < 0 ? -vabsstep : vabsstep
    update: pos -= vstep
    read: magnitude[0]
    write: nx[p] = pos < 0.0 ? pos + magnitude[0] : pos >= magnitude[0] ? pos - magnitude[0] : pos
    read: direction[1]
    compute: double pos = direction[1]
    read: vy[p]
    compute: double value = vy[p]
    compute: double vabs = value < 0 ? -value : value
    read: magnitude[1]
    compute: double vabsstep = (vabs * step) - ((int64_t)(vabs * step) / magnitude[1]) * magnitude[1]
    compute: double vstep = value < 0 ? -vabsstep : vabsstep
    update: pos -= vstep
    read: magnitude[1]
    write: ny[p] = pos < 0.0 ? pos + magnitude[1] : pos >= magnitude[1] ? pos - magnitude[1] : pos
    read: direction[2]
    compute: double pos = direction[2]
    read: vz[p]
    compute: double value = vz[p]
    compute: double vabs = value < 0 ? -value : value
    read: magnitude[2]
    compute: double vabsstep = (vabs * step) - ((int64_t)(vabs * step) / magnitude[2]) * magnitude[2]
    compute: double vstep = value < 0 ? -vabsstep : vabsstep
    update: pos -= vstep
    read: magnitude[2]
    write: nz[p] = pos < 0.0 ? pos + magnitude[2] : pos >= magnitude[2] ? pos - magnitude[2] : pos
```

### arm_simd_loops.loop_019

- route: RSB_pseudocode / generic_code_like_pseudocode
- features: aos_struct_field, array_write, loop_nest

```text
void arm_simd_loop_019(object_t * objects, uint32_t * indexes, int64_t n):
  for (size_t base = 0; base < n; base += VL<object_t>):
    pg = active_lanes<object_t>(base, n)
    lane int i = base + lane_id where pg
    read: indexes[i]
    write: objects[indexes[i]].mark = 1
```

### arm_simd_loops.loop_022

- route: RSB_pseudocode / reduction_or_two_pass
- features: bitwise_or_shift, loop_nest, nested_loop, predicate_or_select_map, reduction_or_accumulator

```text
uint32_t arm_simd_loop_022(uint8_t * p, uint8_t * lmt):
  init: uint32_t res = 0
  while p < lmt:
    compute: uint16_t * plength = (uint16_t *)(p + 1)
    compute: uint16_t length = *plength & 0xfe
    init: uint64_t sum = 0
    compute: uint8_t * tcp_lmt = p + length
    if tcp_lmt > lmt:
      update: tcp_lmt = lmt
    for (uint8_t *q = p; q + 1 < tcp_lmt; q += 2):
      update: memcpy(&word, q, sizeof(word))
      update: sum += word
    update: uint16_t checksum = (uint16_t)~((sum & 0xffffu) + (sum >> 16))
    compute: uint16_t advance = *plength
    if advance == 0:
      break
    update: p += advance
    update: res += 1
    update: res ^= checksum << 16
  return: res
```

### arm_simd_loops.loop_023

- route: RSB_pseudocode / reduction_or_two_pass
- features: array_read, loop_nest, reduction_or_accumulator

```text
double arm_simd_loop_023(double * a, double * b, uint32_t * indexes, int n):
  init: double res = 0
  for (size_t base = 0; base < n; base += VL<double>):
    pg = active_lanes<double>(base, n)
    lane int i = base + lane_id where pg
    read: a[indexes[i]], b[i], indexes[i]
    reduce: res = res + a[indexes[i]] * b[i]
  return: res
```

### arm_simd_loops.loop_024

- route: RSB_pseudocode / reduction_or_two_pass
- features: array_read, loop_nest, reduction_or_accumulator, scalar_reduction_return

```text
uint32_t arm_simd_loop_024(uint8_t * a, uint8_t * b, int64_t n):
  init: uint32_t sum = 0
  for (size_t base = 0; base < n; base += VL<uint8_t>):
    pg = active_lanes<uint8_t>(base, n)
    lane int i = base + lane_id where pg
    read: a[i], b[i]
    reduce: sum += __builtin_abs(a[i] - b[i])
  return: sum
```

### arm_simd_loops.loop_025

- route: RSB_pseudocode / store_map
- features: array_read, array_write, loop_nest, nested_loop, strided_or_flattened_layout

```text
void arm_simd_loop_025(float * a, float * b, float * c):
  for (size_t base = 0; base < 8; base += VL<int>):
    pg = active_lanes<int>(base, 8)
    lane int m = base + lane_id where pg
    compute: int offset = m * 8 * 8
    compute: float * ma = a + offset
    compute: float * mb = b + offset
    compute: float * mc = c + offset
    for (int row = 0; row < 8; row++):
      for (int col = 0; col < 8; col++):
        write: mc[col + row * 8] = 0.0f
        for (int i = 0; i < 8; i++):
          read: ma[i + row * 8], mb[col + i * 8], mc[col + row * 8]
          reduce: mc[col + row * 8] += ma[i + row * 8] * mb[col + i * 8]
```

### arm_simd_loops.loop_026

- route: RSB_pseudocode / predicate_select_store_map
- features: array_read, array_write, bitwise_or_shift, control_or_hybrid, loop_nest, nested_loop, predicate_or_select_map

```text
void arm_simd_loop_026(uint16_t * p, uint8_t * d, uint16_t * lmt, uint16_t * table1, uint16_t * table2, uint16_t * table3):
  while p < lmt:
    read: p[0]
    compute: uint16_t length = p[0]
    if length == 0:
      break
    for (uint16_t conv_i = 0; conv_i < length && p + conv_i < lmt; conv_i++):
      read: p[conv_i]
      compute: uint32_t raw = p[conv_i]
      read: table1[raw >> 10]
      compute: uint32_t first = table1[raw >> 10]
      update: first += (raw >> 4) & 0x3fu
      read: table2[first]
      compute: uint32_t second = table2[first]
      update: second += raw & 0xfu
      read: table3[second]
      compute: uint32_t result = table3[second]
      if result >= 0x100u:
        break
      write: d[conv_i] = (uint8_t)result
    update: p += length
    update: d += length
```

### arm_simd_loops.loop_027

- route: RSB_pseudocode / generic_code_like_pseudocode
- features: array_read, array_write, loop_nest

```text
void arm_simd_loop_027(float * input, float * output, int64_t size):
  for (size_t base = 0; base < size; base += VL<float>):
    pg = active_lanes<float>(base, size)
    lane int64_t i = base + lane_id where pg
    read: input[i]
    write: output[i] = __builtin_sqrtf(input[i])
```

### arm_simd_loops.loop_028

- route: RSB_pseudocode / generic_code_like_pseudocode
- features: array_read, array_write, loop_nest

```text
void arm_simd_loop_028(double * input1, double * input2, double * output, int64_t size):
  for (size_t base = 0; base < size; base += VL<double>):
    pg = active_lanes<double>(base, size)
    lane int64_t i = base + lane_id where pg
    read: input1[i], input2[i]
    write: output[i] = input1[i] / input2[i]
```

### arm_simd_loops.loop_029

- route: RSB_pseudocode / generic_code_like_pseudocode
- features: array_read, array_write, loop_nest

```text
void arm_simd_loop_029(double * input, int64_t * scale, double * output, int64_t size):
  for (size_t base = 0; base < size; base += VL<double>):
    pg = active_lanes<double>(base, size)
    lane int64_t i = base + lane_id where pg
    read: input[i], scale[i]
    write: output[i] = __builtin_scalbn(input[i], (int)scale[i])
```

### arm_simd_loops.loop_031

- route: RSB_pseudocode / reduction_or_two_pass
- features: array_read, loop_nest, nested_loop, reduction_or_accumulator

```text
void arm_simd_loop_031(uint8_t * a, uint8_t * b):
  compute: const std::size_t[20] count = {0, 1, 2, 3, 4, 5, 6, 7, 8, 15, 16, 31, 64, 80, 96, 127, 128, 200, 255, 512}
  init: uint8_t * src = a
  init: uint8_t * to = b
  for (int j = 0; j < 10; j++):
    for (int c = 0; c < CHUNKS; c++):
      update: memcpy(to, src, count[c])
      read: count[c]
      update: src += count[c]
      read: count[c]
      update: to += count[c]
```
