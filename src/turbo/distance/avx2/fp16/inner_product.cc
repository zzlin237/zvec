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

// This file is compiled with per-file -march=core-avx2 (set in
// CMakeLists.txt), which also enables the F16C vcvtph2ps conversion used
// here. When the build toolchain cannot emit AVX2 code, each function falls
// back to a no-op stub guarded by #if defined(__AVX2__).
//
// Kernel ported from ailego/math/inner_product_matrix_fp16_avx.cc:
// halves are widened to fp32 with vcvtph2ps and accumulated with 2x-unrolled
// dual accumulators; the tail is handled through a zero-padded staging
// buffer (zero lanes contribute nothing to the dot product).

#include "avx2/fp16/inner_product.h"
#if defined(__AVX2__)
#include <immintrin.h>
#include "common/hsum_common.h"
#endif
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace zvec::turbo::avx2 {

void inner_product_fp16_distance_avx2(const void *a, const void *b, size_t dim,
                                      float *distance) {
#if defined(__AVX2__)
  const uint16_t *lhs = reinterpret_cast<const uint16_t *>(a);
  const uint16_t *rhs = reinterpret_cast<const uint16_t *>(b);
  const uint16_t *last = lhs + dim;
  const uint16_t *last_aligned = lhs + ((dim >> 4) << 4);

  __m256 ymm_sum_0 = _mm256_setzero_ps();
  __m256 ymm_sum_1 = _mm256_setzero_ps();

  if ((reinterpret_cast<uintptr_t>(lhs) & 0xf) == 0 &&
      (reinterpret_cast<uintptr_t>(rhs) & 0xf) == 0) {
    for (; lhs != last_aligned; lhs += 16, rhs += 16) {
      ymm_sum_0 = _mm256_fmadd_ps(
          _mm256_cvtph_ps(
              _mm_load_si128(reinterpret_cast<const __m128i *>(lhs + 0))),
          _mm256_cvtph_ps(
              _mm_load_si128(reinterpret_cast<const __m128i *>(rhs + 0))),
          ymm_sum_0);
      ymm_sum_1 = _mm256_fmadd_ps(
          _mm256_cvtph_ps(
              _mm_load_si128(reinterpret_cast<const __m128i *>(lhs + 8))),
          _mm256_cvtph_ps(
              _mm_load_si128(reinterpret_cast<const __m128i *>(rhs + 8))),
          ymm_sum_1);
    }

    if (last >= last_aligned + 8) {
      ymm_sum_0 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_load_si128(
                                      reinterpret_cast<const __m128i *>(lhs))),
                                  _mm256_cvtph_ps(_mm_load_si128(
                                      reinterpret_cast<const __m128i *>(rhs))),
                                  ymm_sum_0);
      lhs += 8;
      rhs += 8;
    }
  } else {
    for (; lhs != last_aligned; lhs += 16, rhs += 16) {
      ymm_sum_0 = _mm256_fmadd_ps(
          _mm256_cvtph_ps(
              _mm_loadu_si128(reinterpret_cast<const __m128i *>(lhs + 0))),
          _mm256_cvtph_ps(
              _mm_loadu_si128(reinterpret_cast<const __m128i *>(rhs + 0))),
          ymm_sum_0);
      ymm_sum_1 = _mm256_fmadd_ps(
          _mm256_cvtph_ps(
              _mm_loadu_si128(reinterpret_cast<const __m128i *>(lhs + 8))),
          _mm256_cvtph_ps(
              _mm_loadu_si128(reinterpret_cast<const __m128i *>(rhs + 8))),
          ymm_sum_1);
    }

    if (last >= last_aligned + 8) {
      ymm_sum_0 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128(
                                      reinterpret_cast<const __m128i *>(lhs))),
                                  _mm256_cvtph_ps(_mm_loadu_si128(
                                      reinterpret_cast<const __m128i *>(rhs))),
                                  ymm_sum_0);
      lhs += 8;
      rhs += 8;
    }
  }

  ymm_sum_0 = _mm256_add_ps(ymm_sum_0, ymm_sum_1);
  if (lhs != last) {
    // Zero-padded staging buffers: padded lanes yield product 0.
    alignas(16) uint16_t buf_m[8] = {0};
    alignas(16) uint16_t buf_q[8] = {0};
    size_t remain = static_cast<size_t>(last - lhs);
    memcpy(buf_m, lhs, remain * sizeof(uint16_t));
    memcpy(buf_q, rhs, remain * sizeof(uint16_t));
    ymm_sum_0 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_load_si128(
                                    reinterpret_cast<const __m128i *>(buf_m))),
                                _mm256_cvtph_ps(_mm_load_si128(
                                    reinterpret_cast<const __m128i *>(buf_q))),
                                ymm_sum_0);
  }

  *distance = -internal::hsum_fp32_v256(ymm_sum_0);
#else
  (void)a;
  (void)b;
  (void)dim;
  (void)distance;
#endif
}

void inner_product_fp16_batch_distance_avx2(const void *const *vectors,
                                            const void *query, size_t n,
                                            size_t dim, float *distances) {
#if defined(__AVX2__)
  for (size_t i = 0; i < n; ++i) {
    inner_product_fp16_distance_avx2(vectors[i], query, dim, &distances[i]);
  }
#else
  (void)vectors;
  (void)query;
  (void)n;
  (void)dim;
  (void)distances;
#endif
}

}  // namespace zvec::turbo::avx2
