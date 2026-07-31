// Copyright 2025-present the zvec project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// This file is compiled with per-file -march for AVX512 (set in
// CMakeLists.txt). When the build toolchain cannot emit AVX512 code, each
// function falls back to a no-op stub guarded by #if defined(__AVX512F__).
//
// Kernel ported from ailego/math/euclidean_distance_matrix_fp16_avx512.cc:
// halves are widened to fp32 with vcvtph2ps (AVX512F, no AVX512FP16 needed)
// and accumulated with 2x-unrolled dual accumulators; the tail is handled
// through a zero-padded staging buffer (zero lanes contribute nothing to the
// squared difference).

#include "avx512/fp16/squared_euclidean.h"
#if defined(__AVX512F__)
#include <immintrin.h>
#include "common/hsum_common.h"
#endif
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace zvec::turbo::avx512 {

void squared_euclidean_fp16_distance_avx512(const void *a, const void *b,
                                            size_t dim, float *distance) {
#if defined(__AVX512F__)
  const uint16_t *lhs = reinterpret_cast<const uint16_t *>(a);
  const uint16_t *rhs = reinterpret_cast<const uint16_t *>(b);
  const uint16_t *last = lhs + dim;
  const uint16_t *last_aligned = lhs + ((dim >> 5) << 5);

  __m512 zmm_sum_0 = _mm512_setzero_ps();
  __m512 zmm_sum_1 = _mm512_setzero_ps();

  if ((reinterpret_cast<uintptr_t>(lhs) & 0x3f) == 0 &&
      (reinterpret_cast<uintptr_t>(rhs) & 0x3f) == 0) {
    for (; lhs != last_aligned; lhs += 32, rhs += 32) {
      __m512 zmm_d_0 = _mm512_sub_ps(
          _mm512_cvtph_ps(
              _mm256_load_si256(reinterpret_cast<const __m256i *>(lhs + 0))),
          _mm512_cvtph_ps(
              _mm256_load_si256(reinterpret_cast<const __m256i *>(rhs + 0))));
      __m512 zmm_d_1 = _mm512_sub_ps(
          _mm512_cvtph_ps(
              _mm256_load_si256(reinterpret_cast<const __m256i *>(lhs + 16))),
          _mm512_cvtph_ps(
              _mm256_load_si256(reinterpret_cast<const __m256i *>(rhs + 16))));
      zmm_sum_0 = _mm512_fmadd_ps(zmm_d_0, zmm_d_0, zmm_sum_0);
      zmm_sum_1 = _mm512_fmadd_ps(zmm_d_1, zmm_d_1, zmm_sum_1);
    }

    if (last >= last_aligned + 16) {
      __m512 zmm_d = _mm512_sub_ps(
          _mm512_cvtph_ps(
              _mm256_load_si256(reinterpret_cast<const __m256i *>(lhs))),
          _mm512_cvtph_ps(
              _mm256_load_si256(reinterpret_cast<const __m256i *>(rhs))));
      zmm_sum_0 = _mm512_fmadd_ps(zmm_d, zmm_d, zmm_sum_0);
      lhs += 16;
      rhs += 16;
    }
  } else {
    for (; lhs != last_aligned; lhs += 32, rhs += 32) {
      __m512 zmm_d_0 = _mm512_sub_ps(
          _mm512_cvtph_ps(
              _mm256_loadu_si256(reinterpret_cast<const __m256i *>(lhs + 0))),
          _mm512_cvtph_ps(
              _mm256_loadu_si256(reinterpret_cast<const __m256i *>(rhs + 0))));
      __m512 zmm_d_1 = _mm512_sub_ps(
          _mm512_cvtph_ps(
              _mm256_loadu_si256(reinterpret_cast<const __m256i *>(lhs + 16))),
          _mm512_cvtph_ps(
              _mm256_loadu_si256(reinterpret_cast<const __m256i *>(rhs + 16))));
      zmm_sum_0 = _mm512_fmadd_ps(zmm_d_0, zmm_d_0, zmm_sum_0);
      zmm_sum_1 = _mm512_fmadd_ps(zmm_d_1, zmm_d_1, zmm_sum_1);
    }

    if (last >= last_aligned + 16) {
      __m512 zmm_d = _mm512_sub_ps(
          _mm512_cvtph_ps(
              _mm256_loadu_si256(reinterpret_cast<const __m256i *>(lhs))),
          _mm512_cvtph_ps(
              _mm256_loadu_si256(reinterpret_cast<const __m256i *>(rhs))));
      zmm_sum_0 = _mm512_fmadd_ps(zmm_d, zmm_d, zmm_sum_0);
      lhs += 16;
      rhs += 16;
    }
  }

  zmm_sum_0 = _mm512_add_ps(zmm_sum_0, zmm_sum_1);
  if (lhs != last) {
    // Zero-padded staging buffers: padded lanes yield diff 0, adding nothing.
    alignas(32) uint16_t buf_m[16] = {0};
    alignas(32) uint16_t buf_q[16] = {0};
    size_t remain = static_cast<size_t>(last - lhs);
    memcpy(buf_m, lhs, remain * sizeof(uint16_t));
    memcpy(buf_q, rhs, remain * sizeof(uint16_t));
    __m512 zmm_d = _mm512_sub_ps(
        _mm512_cvtph_ps(
            _mm256_load_si256(reinterpret_cast<const __m256i *>(buf_m))),
        _mm512_cvtph_ps(
            _mm256_load_si256(reinterpret_cast<const __m256i *>(buf_q))));
    zmm_sum_0 = _mm512_fmadd_ps(zmm_d, zmm_d, zmm_sum_0);
  }

  *distance = internal::hsum_fp32_v512(zmm_sum_0);
#else
  (void)a;
  (void)b;
  (void)dim;
  (void)distance;
#endif
}

void squared_euclidean_fp16_batch_distance_avx512(const void *const *vectors,
                                                  const void *query, size_t n,
                                                  size_t dim,
                                                  float *distances) {
#if defined(__AVX512F__)
  for (size_t i = 0; i < n; ++i) {
    squared_euclidean_fp16_distance_avx512(vectors[i], query, dim,
                                           &distances[i]);
  }
#else
  (void)vectors;
  (void)query;
  (void)n;
  (void)dim;
  (void)distances;
#endif
}

}  // namespace zvec::turbo::avx512
