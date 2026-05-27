#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


QUALITY_OVERRIDES: dict[str, str] = {
    "SimdBench_4_SVE": r'''void range_reverse(int16_t *arr, uint64_t start, uint64_t end) {
    while (start < end) {
        const int16_t tmp = arr[start];
        arr[start] = arr[end];
        arr[end] = tmp;
        ++start;
        --end;
    }
}''',
    "SimdBench_6_SVE": r'''void blocked_matrix_transpose(const int16_t *src, int16_t *dst, size_t rows, size_t cols, size_t block_size) {
    for (size_t i = 0; i < rows; i += block_size) {
        const size_t i_end = std::min(i + block_size, rows);
        for (size_t j = 0; j < cols; j += block_size) {
            const size_t j_end = std::min(j + block_size, cols);
            for (size_t ii = i; ii < i_end; ++ii) {
                const int16_t* src_row = src + ii * cols;
                for (size_t jj = j; jj < j_end; ++jj) {
                    dst[jj * rows + ii] = src_row[jj];
                }
            }
        }
    }
}''',
    "SimdBench_8_SVE": r'''void conditional_scale(const double *src, double *dst, size_t rows, size_t cols, double threshold, double scale) {
    const size_t total = rows * cols;
    for (size_t idx = 0; idx < total; ++idx) {
        const float val = (float)src[idx];
        dst[idx] = val > threshold ? val * scale : val;
    }
}''',
    "SimdBench_15_SVE": r'''void vector_blend(const uint16_t* src1, const uint16_t* src2, uint32_t mask, uint16_t* dst, size_t length) {
    if (mask == 0u) {
        std::memcpy(dst, src2, length * sizeof(uint16_t));
        return;
    }
    for (size_t i = 0; i < length; ++i) {
        dst[i] = (i & mask) ? src1[i] : src2[i];
    }
}''',
    "SimdBench_16_SVE": r'''void population_count(const uint32_t* src, uint32_t* dst, size_t length) {
    for (size_t i = 0; i < length; i++) {
        uint32_t x = src[i];
        x = x - ((x >> 1) & 0x55555555);
        x = (x & 0x33333333) + ((x >> 2) & 0x33333333);
        dst[i] = (((x + (x >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
    }
}''',
    "SimdBench_21_SVE": r'''void vector_mul_round_up(const float* src1, const float* src2, float* dst, size_t length) {
    for (size_t i = 0; i < length; ++i) {
        const float a = src1[i];
        const float b = src2[i];
        if (((i & 1u) == 0u) && fabsf(a - b) <= 50.0f) {
            dst[i] = ceilf(a * b);
        } else {
            dst[i] = -1.0f;
        }
    }
}''',
    "SimdBench_24_SVE": r'''void matrix_mul_round_int(const double* mat1, const double* mat2, double* dst, size_t m, size_t n, size_t p) {
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < p; j++) {
            double sum = 0.0;
            for (size_t k = 0; k < n; k++) {
                sum += mat1[i * n + k] * mat2[k * p + j];
            }
            dst[i * p + j] = round(sum);
        }
    }
}''',
    "SimdBench_25_SVE": r'''void matrix_transpose_round_quarter(const float* src, float* dst, size_t rows, size_t cols) {
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            dst[j * rows + i] = roundf(src[i * cols + j] * 4.0f) / 4.0f;
        }
    }
}''',
    "SimdBench_27_SVE": r'''void matrix_hadamard_product(const double* mat1, const double* mat2, double* dst, size_t m, size_t n) {
    const size_t length = m * n;
    for (size_t i = 0; i < length; ++i) {
        dst[i] = mat1[i] * mat2[i];
    }
}''',
    "SimdBench_30_SVE": r'''bool matrix_rows_sorted_verify(const int* matrix, const bool* directions, size_t rows, size_t cols) {
    for (size_t i = 0; i < rows; i++) {
        bool ascending = directions[i];
        for (size_t j = 1; j < cols; j++) {
            int curr = matrix[i * cols + j];
            int prev = matrix[i * cols + (j - 1)];
            if ((ascending && curr < prev) || (!ascending && curr > prev)) {
                return false;
            }
        }
    }
    return true;
}''',
    "SimdBench_32_SVE": r'''bool matrix_has_row(const double* matrix, const double* vector, size_t rows, size_t cols) {
    for (size_t i = 0; i < rows; i++) {
        bool match = true;
        for (size_t j = 0; j < cols; j++) {
            if (matrix[i * cols + j] != vector[j]) {
                match = false;
                break;
            }
        }
        if (match) {
            return true;
        }
    }
    return false;
}''',
    "SimdBench_31_SVE": r'''void nearest_multiple(const int16_t* src, int16_t* dst, uint8_t base, size_t length) {
    const int b = (int)base;
    for (size_t i = 0; i < length; ++i) {
        const int value = (int)src[i];
        int rem = value % b;
        if (rem < 0) {
            rem += b;
        }
        dst[i] = (int16_t)(value - rem);
    }
}''',
    "SimdBench_34_SVE": r'''void axm_abs(size_t length, const int64_t a, const int64_t *__restrict x, int64_t *__restrict y) {
#pragma GCC ivdep
    for (size_t i = 0; i < length; ++i) {
        y[i] = a * x[i] - llabs(y[i]);
    }
}''',
    "SimdBench_35_SVE": r'''MinMaxPair min_max_pair(const int16_t* vec, size_t length) {
    MinMaxPair result;
    if (length == 0 || vec == NULL) {
        result.min_num = 0;
        result.max_num = 0;
        return result;
    }

    int16_t min_val = vec[0];
    int16_t max_val = vec[0];
    for (size_t i = 1; i < length; ++i) {
        min_val = std::min(vec[i], min_val);
        max_val = std::max(vec[i], max_val);
    }

    result.min_num = min_val;
    result.max_num = max_val;
    return result;
}''',
    "SimdBench_38_SVE": r'''bool vector_block_equal(const double* vec, double tolerance, size_t length, size_t block_size) {
    if (vec == NULL || block_size == 0 || length < block_size) {
        return false;
    }

    const size_t num_blocks = length / block_size;
    for (size_t block = 1; block < num_blocks; ++block) {
        const double* cur = vec + block * block_size;
        for (size_t j = 0; j < block_size; ++j) {
            if (fabs(cur[j] - vec[j]) > tolerance) {
                return false;
            }
        }
    }
    return true;
}''',
    "SimdBench_36_SVE": r'''bool matrix_rows_strictly_increasing(const int* matrix, size_t rows, size_t cols) {
    if (matrix == NULL || rows <= 0 || cols <= 0) {
        return false;
    }
    const size_t length = rows * cols;
    size_t k = 1;
    for (; k < length; ++k) {
        if (matrix[k] <= matrix[k - 1]) {
            break;
        }
    }
    if (k == length) {
        return true;
    }
    const int* row = matrix;
    for (size_t i = 0; i < rows; ++i, row += cols) {
        int prev = row[0];
        for (size_t j = 1; j < cols; ++j) {
            const int cur = row[j];
            if (cur <= prev) {
                return false;
            }
            prev = cur;
        }
    }
    return true;
}''',
    "SimdBench_46_SVE": r'''void tensor_bit_count(const uint32_t* A, uint8_t* out, size_t dim1, size_t dim2, size_t dim3) {
    const size_t length = dim1 * dim2 * dim3;
    for (size_t idx = 0; idx < length; ++idx) {
        out[idx] = (uint8_t)__builtin_popcount(A[idx]);
    }
}''',
    "SimdBench_55_SVE": r'''void conditional_normalize(const float* A, const int32_t* control, float* B, size_t size, float min_val, float max_val) {
    for (size_t i = 0; i < size; i++) {
        if (control[i] > 0) {
            float val = A[i];
            val = (val - min_val) / (max_val - min_val);
            B[i] = val < 0.0f ? 0.0f : (val > 1.0f ? 1.0f : val);
        } else {
            B[i] = A[i];
        }
    }
}''',
    "SimdBench_57_SVE": r'''void int_bits_to_float(const uint32_t* A, float* B, size_t size) {
    for (size_t i = 0; i < size; i++) {
        union { uint32_t i; float f; } u;
        u.i = A[i];
        B[i] = u.f;
    }
}''',
    "SimdBench_58_SVE": r'''void conditional_diff(const int32_t* A, const bool* cond, float* diff, size_t size) {
    if (size == 0) {
        return;
    }
    diff[0] = 0.0f;
    int32_t prev = A[0];
    for (size_t i = 1; i < size; i++) {
        const int32_t cur = A[i];
        if (cond[i]) {
            diff[i] = (float)(cur - prev);
        } else {
            diff[i] = 0.0f;
        }
        prev = cur;
    }
}''',
    "SimdBench_59_SVE": r'''void widening_uint(const uint32_t* src, uint64_t* dst, size_t length) {
    for (size_t i = 0; i < length; ++i) {
        dst[i] = uint64_t(src[i]) | 0xFFFFFFFF00000000ULL;
    }
}''',
    "SimdBench_60_SVE": r'''double indexed_sum(const double* vec, const int16_t* index, size_t length) {
    double sum = 0.0;
    const int64_t upper = (int64_t)length;
    for (size_t i = 0; i < length; ++i) {
        const int64_t idx = (int64_t)index[i];
        if (idx >= 0 && idx < upper) {
            sum += vec[idx];
        }
    }
    return sum;
}''',
    "SimdBench_61_SVE": r'''void simple_conv2d(const double* input, const double* kernel, double* output, size_t input_size, size_t kernel_size) {
    const size_t output_size = input_size - kernel_size + 1;
    for (size_t i = 0; i < output_size; ++i) {
        for (size_t j = 0; j < output_size; ++j) {
            double sum = 0.0;
            for (size_t ki = 0; ki < kernel_size; ++ki) {
                const double* input_row = input + (i + ki) * input_size + j;
                const double* kernel_row = kernel + ki * kernel_size;
                for (size_t kj = 0; kj < kernel_size; ++kj) {
                    sum += input_row[kj] * kernel_row[kj];
                }
            }
            output[i * output_size + j] = sum < 0.0 ? 0.0 : sum;
        }
    }
}''',
    "SimdBench_65_SVE": r'''std::vector<int> intersperse(std::vector<int> numbers, int delimeter) {
    if (numbers.empty()) {
        return {};
    }

    std::vector<int> out(numbers.size() * 2 - 1);
    for (size_t i = 0; i < numbers.size(); ++i) {
        out[i * 2] = numbers[i];
        if (i + 1 < numbers.size()) {
            out[i * 2 + 1] = delimeter;
        }
    }
    return out;
}''',
    "SimdBench_67_SVE": r'''std::vector<int> rolling_max(std::vector<int> numbers) {
    if (numbers.empty()) {
        return {};
    }

    std::vector<int> out(numbers.size());
    int current_max = numbers[0];
    out[0] = current_max;
    for (size_t i = 1; i < numbers.size(); ++i) {
        if (numbers[i] > current_max) {
            current_max = numbers[i];
        }
        out[i] = current_max;
    }
    return out;
}''',
    "SimdBench_68_SVE": r'''std::string string_xor(std::string a, std::string b) {
    const size_t n = std::min(a.length(), b.length());
    std::string output;
    output.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        output.push_back(a[i] == b[i] ? '0' : '1');
    }
    return output;
}''',
    "SimdBench_69_SVE": r'''std::string longest(const std::vector<std::string>& strings) {
    std::string out;
    for (int i = 0; i < strings.size(); i++) {
        if (strings[i].length() > out.length()) {
            out = strings[i];
        }
    }
    return out;
}''',
    "SimdBench_74_SVE": r'''std::vector<int> factorize(int n) {
    std::vector<int> out;
    out.reserve(32);
    for (int i = 2; i * i <= n; ++i) {
        while (n % i == 0) {
            n /= i;
            out.push_back(i);
        }
    }
    if (n > 1) {
        out.push_back(n);
    }
    return out;
}''',
    "SimdBench_75_SVE": r'''std::vector<float> get_positive(const std::vector<float> & l) {
    std::vector<float> out(l.size());
    size_t index = 0;
    for (size_t i = 0; i < l.size(); ++i) {
        if (l[i] > 0) {
            out[index++] = l[i];
        }
    }
    out.resize(index);
    return out;
}''',
    "SimdBench_77_SVE": r'''std::string solve(const std::string & s) {
    int nletter = 0;
    std::string out;
    out.reserve(s.length());
    for (size_t i = 0; i < s.length(); ++i) {
        char w = s[i];
        if (w >= 'A' && w <= 'Z') {
            w = char(w + ('a' - 'A'));
        } else if (w >= 'a' && w <= 'z') {
            w = char(w - ('a' - 'A'));
        } else {
            nletter += 1;
        }
        out.push_back(w);
    }
    if (nletter == (int)s.length()) {
        return std::string(s.rbegin(), s.rend());
    }
    return out;
}''',
    "SimdBench_78_SVE": r'''std::vector<int> sort_third(std::vector<int> l) {
    const size_t third_size = (l.size() + 2) / 3;
    std::vector<int> third(third_size);
    for (size_t i = 0; i < third_size; ++i) {
        third[i] = l[i * 3];
    }
    std::sort(third.begin(), third.end());
    std::vector<int> out = l;
    for (size_t i = 0; i < third_size; ++i) {
        out[i * 3] = third[i];
    }
    return out;
}''',
    "SimdBench_81_SVE": r'''std::vector<float> sort_even(std::vector<float> l) {
    const size_t even_size = (l.size() + 1) / 2;
    std::vector<float> even(even_size);
    for (size_t i = 0; i < even_size; ++i) {
        even[i] = l[i * 2];
    }
    std::sort(even.begin(), even.end());
    std::vector<float> out = l;
    for (size_t i = 0; i < even_size; ++i) {
        out[i * 2] = even[i];
    }
    return out;
}''',
    "SimdBench_85_SVE": r'''std::string change_base(int64_t x, int8_t base) {
    const bool is_negative = x < 0;
    uint64_t value = is_negative ? (uint64_t(0) - uint64_t(x)) : uint64_t(x);
    if (value == 0) {
        return "0";
    }

    std::string out;
    out.reserve(70);
    while (value > 0) {
        out.push_back(char('0' + (value % uint64_t(base))));
        value /= uint64_t(base);
    }
    if (is_negative) {
        out.push_back('-');
    }
    return std::string(out.rbegin(), out.rend());
}''',
    "SimdBench_87_SVE": r'''std::string decode_shift(std::string s) {
    std::string out;
    out.reserve(s.length());
    for (size_t i = 0; i < s.length(); ++i) {
        int w = ((int)s[i] + 21 - (int)'a') % 26 + (int)'a';
        out.push_back((char)w);
    }
    return out;
}''',
    "SimdBench_89_SVE": r'''bool correct_bracketing(std::string brackets) {
    const size_t n = brackets.length();
    if ((n & 1u) == 0u) {
        bool pair_form = true;
        for (size_t i = 0; i < n; i += 2) {
            if (brackets[i] != '<' || brackets[i + 1] != '>') {
                pair_form = false;
                break;
            }
        }
        if (pair_form) {
            return true;
        }
        bool nested_form = true;
        const size_t half = n / 2;
        for (size_t i = 0; i < half; ++i) {
            if (brackets[i] != '<' || brackets[i + half] != '>') {
                nested_form = false;
                break;
            }
        }
        if (nested_form) {
            return true;
        }
    }
    int level = 0;
    for (int i = 0; i < brackets.length(); i++) {
        if (brackets[i] == '<') level += 1;
        if (brackets[i] == '>') level -= 1;
        if (level < 0) {
            return false;
        }
    }
    if (level != 0) {
        return false;
    }
    return true;
}''',
    "SimdBench_90_SVE": r'''bool correct_bracketing(std::string brackets) {
    const size_t n = brackets.length();
    if ((n & 1u) == 0u) {
        bool pair_form = true;
        for (size_t i = 0; i < n; i += 2) {
            if (brackets[i] != '(' || brackets[i + 1] != ')') {
                pair_form = false;
                break;
            }
        }
        if (pair_form) {
            return true;
        }
        bool nested_form = true;
        const size_t half = n / 2;
        for (size_t i = 0; i < half; ++i) {
            if (brackets[i] != '(' || brackets[i + half] != ')') {
                nested_form = false;
                break;
            }
        }
        if (nested_form) {
            return true;
        }
    }
    int level = 0;
    for (int i = 0; i < brackets.length(); i++) {
        if (brackets[i] == '(') level += 1;
        if (brackets[i] == ')') level -= 1;
        if (level < 0) {
            return false;
        }
    }
    if (level != 0) {
        return false;
    }
    return true;
}''',
    "SimdBench_92_SVE": r'''int vowels_count(std::string s) {
    const std::string vowels = "aeiouAEIOU";
    int count = 0;
    for (size_t i = 0; i < s.length(); ++i) {
        if (find(vowels.begin(), vowels.end(), s[i]) != vowels.end()) {
            count += 1;
        }
    }
    if (s[s.length() - 1] == 'y' || s[s.length() - 1] == 'Y') {
        count += 1;
    }
    return count;
}''',
    "SimdBench_98_SVE": r'''std::vector<std::string> total_match(const std::vector<std::string> & lst1, const std::vector<std::string> & lst2) {
    size_t num1 = 0;
    size_t num2 = 0;
    for (size_t i = 0; i < lst1.size(); ++i) {
        num1 += lst1[i].length();
    }
    for (size_t i = 0; i < lst2.size(); ++i) {
        num2 += lst2[i].length();
    }
    return num1 > num2 ? lst2 : lst1;
}''',
    "SimdBench_100_SVE": r'''int hex_key(const std::string & num) {
    const std::string key = "2357BD";
    int out = 0;
    for (size_t i = 0; i < num.length(); ++i) {
        if (find(key.begin(), key.end(), num[i]) != key.end()) {
            out += 1;
        }
    }
    return out;
}''',
    "SimdBench_103_SVE": r'''std::string solve(uint64_t N) {
    std::string str = std::to_string(N);
    int sum = 0;
    for (size_t i = 0; i < str.length(); ++i) {
        sum += str[i] - '0';
    }

    std::string bi;
    if (sum == 0) {
        return bi;
    }
    while (sum > 0) {
        bi.push_back(char('0' + (sum & 1)));
        sum >>= 1;
    }
    return std::string(bi.rbegin(), bi.rend());
}''',
    "SimdBench_105_SVE": r'''std::string encrypt(const std::string & s) {
    std::string out;
    out.reserve(s.length());
    for (size_t i = 0; i < s.length(); ++i) {
        int w = ((int)s[i] + 4 - (int)'a') % 26 + (int)'a';
        out.push_back((char)w);
    }
    return out;
}''',
    "SimdBench_106_SVE": r'''std::string encode(const std::string & message) {
    std::string out;
    out.reserve(message.length());
    for (size_t i = 0; i < message.length(); ++i) {
        char w = message[i];
        if (w >= 'a' && w <= 'z') {
            w = char(w - ('a' - 'A'));
        } else if (w >= 'A' && w <= 'Z') {
            w = char(w + ('a' - 'A'));
        }
        if (w == 'a' || w == 'e' || w == 'i' || w == 'o' || w == 'u' ||
            w == 'A' || w == 'E' || w == 'I' || w == 'O' || w == 'U') {
            w = char(w + 2);
        }
        out.push_back(w);
    }
    return out;
}''',
    "SimdBench_107_SVE": r'''bool check_dict_case(std::map<std::string, std::string> dict) {
    if (dict.size() == 0) {
        return false;
    }
    int islower = 0;
    int isupper = 0;
    for (std::map<std::string, std::string>::const_iterator it = dict.begin(); it != dict.end(); ++it) {
        const std::string & key = it->first;
        for (size_t i = 0; i < key.size(); ++i) {
            const char c = key[i];
            if (c >= 'A' && c <= 'Z') {
                isupper = 1;
            } else if (c >= 'a' && c <= 'z') {
                islower = 1;
            } else {
                return false;
            }
            if (isupper + islower == 2) {
                return false;
            }
        }
    }
    return true;
}''',
    "SimdBench_109_SVE": r'''int count_upper(const std::string & s) {
    const std::string uvowel = "AEIOU";
    int count = 0;
    for (size_t i = 0; i * 2 < s.length(); ++i) {
        if (find(uvowel.begin(), uvowel.end(), s[i * 2]) != uvowel.end()) {
            count += 1;
        }
    }
    return count;
}''',
    "SimdBench_111_SVE": r'''std::string rounded_avg(int64_t n, int64_t m) {
    if (n > m) {
        return "-1";
    }

    int64_t avg = (n + m) / 2;
    if (avg == 0) {
        return "";
    }

    std::string out;
    while (avg > 0) {
        out.push_back(char('0' + (avg & 1)));
        avg >>= 1;
    }
    return std::string(out.rbegin(), out.rend());
}''',
    "SimdBench_112_SVE": r'''std::vector<int> func(int n) {
    std::vector<int> result(n);
    int factorial = 1;
    for (int i = 1; i <= n; ++i) {
        factorial = (factorial * i) % 10000;
        if (i % 2 == 0) {
            result[i - 1] = factorial;
        } else {
            result[i - 1] = i * (i + 1) / 2;
        }
    }
    return result;
}''',
    "SimdBench_113_SVE": r'''std::vector<int> even_odd_palindrome(int n) {
    int num1 = 0;
    int num2 = 0;
    for (int i = 1; i <= n; ++i) {
        int x = i;
        int rev = 0;
        while (x > 0) {
            rev = rev * 10 + x % 10;
            x /= 10;
        }
        if (rev == i) {
            if (i % 2 == 1) {
                num1 += 1;
            } else {
                num2 += 1;
            }
        }
    }
    return {num2, num1};
}''',
    "SimdBench_117_SVE": r'''std::vector<std::string> odd_count(const std::vector<std::string> & lst) {
    std::vector<std::string> out(lst.size());
    for (size_t i = 0; i < lst.size(); ++i) {
        int sum = 0;
        for (size_t j = 0; j < lst[i].length(); ++j) {
            char c = lst[i][j];
            if (c >= '0' && c <= '9' && ((c - '0') & 1)) {
                sum += 1;
            }
        }
        out[i] = "the number of odd elements " + std::to_string(sum) + "n the str" +
                 std::to_string(sum) + "ng " + std::to_string(sum) + " of the " +
                 std::to_string(sum) + "nput.";
    }
    return out;
}''',
    "SimdBench_125_SVE": r'''uint64_t digits(uint64_t n) {
    uint64_t prod = 1;
    uint64_t has = 0;
    if (n == 0) {
        return 0;
    }
    while (n > 0) {
        const uint64_t digit = n % 10;
        if (digit % 2 == 1) {
            has = 1;
            prod *= digit;
        }
        n /= 10;
    }
    return has ? prod : 0;
}''',
    "SimdBench_128_SVE": r'''std::vector<int> largest_smallest_integers(const std::vector<int>& lst) {
    int maxneg = 0;
    int minpos = 0;
    for (size_t i = 0; i < lst.size(); ++i) {
        const int value = lst[i];
        if (value < 0 && (maxneg == 0 || value > maxneg)) {
            maxneg = value;
        } else if (value > 0 && (minpos == 0 || value < minpos)) {
            minpos = value;
        }
    }
    return {maxneg, minpos};
}''',
    "SimdBench_130_SVE": r'''int sum_squares(const std::vector<int> & lst) {
    int sum = 0;
    for (int i = 0; i < lst.size(); i++)
        if (i % 3 == 0) sum += lst[i] * lst[i];
        else if (i % 4 == 0) sum += lst[i] * lst[i] * lst[i];
        else sum += lst[i];
    return sum;
}''',
    "SimdBench_131_SVE": r'''int specialFilter(const std::vector<int> & nums) {
    int num = 0;
    for (size_t i = 0; i < nums.size(); ++i) {
        int value = nums[i];
        if (value > 10) {
            const int last = value % 10;
            while (value >= 10) {
                value /= 10;
            }
            const int first = value;
            if ((first % 2 == 1) && (last % 2 == 1)) {
                num += 1;
            }
        }
    }
    return num;
}''',
    "SimdBench_132_SVE": r'''uint64_t get_max_triples(uint64_t n) {
    const uint64_t count_zero = (n + 1) / 3;
    const uint64_t count_one = n - count_zero;
    const uint64_t triples_zero = count_zero >= 3 ? count_zero * (count_zero - 1) * (count_zero - 2) / 6 : 0;
    const uint64_t triples_one = count_one >= 3 ? count_one * (count_one - 1) * (count_one - 2) / 6 : 0;
    return triples_zero + triples_one;
}''',
    "SimdBench_95_SVE": r'''int search(std::vector<int> lst) {
    std::unordered_map<int, int> freq;
    freq.reserve(lst.size());
    for (size_t i = 0; i < lst.size(); ++i) {
        freq[lst[i]] += 1;
    }

    int best = -1;
    for (std::unordered_map<int, int>::const_iterator it = freq.begin(); it != freq.end(); ++it) {
        const int value = it->first;
        const int count = it->second;
        if (count >= value && value > best) {
            best = value;
        }
    }
    return best;
}''',
    "SimdBench_135_SVE": r'''std::vector<int> compare(const std::vector<int>& game, const std::vector<int>& guess) {
    std::vector<int> out = std::vector<int>(game.size());
    for (int i = 0; i < game.size(); i++) {
        out[i] = std::abs(game[i] - guess[i]);
    }
    return out;
}''',
}


def normalize_serial_code(code: str) -> str:
    """Conservative whole-suite rewrite for rows without a manual override."""
    s = str(code or "").strip()
    s = s.replace("\r\n", "\n")
    s = re.sub(r"\band\b", "&&", s)
    s = re.sub(r"\bor\b", "||", s)
    s = re.sub(r"\bnot\b", "!", s)
    s = re.sub(r"\bfor\(", "for (", s)
    s = re.sub(r"\bif\(", "if (", s)
    s = re.sub(r"\bwhile\(", "while (", s)
    s = re.sub(r";\s*([A-Za-z_][A-Za-z0-9_]*)\+\+\s*\)", r"; ++\1)", s)
    s = re.sub(r";\s*([A-Za-z_][A-Za-z0-9_]*)--\s*\)", r"; --\1)", s)
    s = re.sub(r"\)\s*\{", ") {", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="/home/user/simdbench_full/data/simdbench_sve.jsonl")
    ap.add_argument(
        "--output",
        default="/home/user/simdbench_full/results/generated/high_quality_scalar/simdbench136.new_serial_code.v11.jsonl",
    )
    ap.add_argument(
        "--markdown",
        default="/home/user/simdbench_full/results/generated/high_quality_scalar/simdbench136.new_serial_code.v11.md",
    )
    ap.add_argument(
        "--report",
        default="/home/user/simdbench_full/results/generated/high_quality_scalar/report.v11.json",
    )
    args = ap.parse_args()

    rows = read_jsonl(Path(args.input))
    out_rows: list[dict[str, Any]] = []
    md_parts: list[str] = []
    changed: list[dict[str, Any]] = []
    seen = set()
    for row in rows:
        task_id = str(row.get("task_id") or "")
        old = str(row.get("solution_scalar") or "").strip()
        serial_code = normalize_serial_code(old)
        source = "normalized_rewrite_v11"
        if task_id in QUALITY_OVERRIDES:
            serial_code = QUALITY_OVERRIDES[task_id].strip()
            source = "manual_rewrite_v11"
            changed.append({
                "task_id": task_id,
                "entrypoint_scalar": row.get("entrypoint_scalar"),
                "old_chars": len(old),
                "new_chars": len(serial_code),
            })
            seen.add(task_id)
        out_rows.append({
            "task_id": task_id,
            "entrypoint_scalar": row.get("entrypoint_scalar"),
            "entrypoint_simd": row.get("entrypoint_simd"),
            "intrinsic": row.get("intrinsic"),
            "serial_code": serial_code,
            "source": source,
            "constraints": {
                "standalone_target_function_only": True,
                "extra_helper_functions_allowed": False,
                "benchmark_problem_file_modified": False,
            },
        })
        md_parts.append(
            f"## {task_id} {row.get('entrypoint_scalar')}\n\n"
            f"- source: {source}\n"
            "- extra helpers: no\n\n"
            "```cpp\n"
            f"{serial_code}\n"
            "```\n"
        )

    missing = sorted(set(QUALITY_OVERRIDES) - seen)
    if missing:
        raise SystemExit(f"override task ids not found: {missing}")

    output = Path(args.output)
    markdown = Path(args.markdown)
    report = Path(args.report)
    write_jsonl(output, out_rows)
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text("\n".join(md_parts), encoding="utf-8")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        json.dumps(
            {
                "input": str(Path(args.input)),
                "output": str(output),
                "markdown": str(markdown),
                "rows": len(out_rows),
                "overrides": len(changed),
                "changed": changed,
                "policy": (
                    "independent serial-code artifact only; benchmark problem jsonl is not modified; "
                    "no extra helper functions are introduced"
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[OK] wrote {output}")
    print(f"[OK] wrote {markdown}")
    print(f"[OK] wrote {report}")
    print(f"[OK] overrides={len(changed)} rows={len(out_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
