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
// CMakeLists.txt). When the build toolchain cannot emit AVX2 code, each
// function falls back to a no-op stub guarded by #if defined(__AVX2__).
//
// Kernel ported from ailego/math/inner_product_matrix_fp32_avx.cc:
// 2x-unrolled dual accumulators, aligned/unaligned dual load paths and a
// scalar tail, adapted to the turbo DistanceFunc signature (negated dot).

#include "avx2/fp32/inner_product.h"
#if defined(__AVX2__)
#include <immintrin.h>
#include "common/hsum_common.h"
#endif
#include <cstddef>
#include <cstdint>

namespace zvec::turbo::avx2 {

void inner_product_fp32_distance_avx2(const void *a, const void *b, size_t dim,
                                      float *distance) {
#if defined(__AVX2__)
  const float *lhs = reinterpret_cast<const float *>(a);
  const float *rhs = reinterpret_cast<const float *>(b);
  const float *last = lhs + dim;
  const float *last_aligned = lhs + ((dim >> 4) << 4);

  __m256 ymm_sum_0 = _mm256_setzero_ps();
  __m256 ymm_sum_1 = _mm256_setzero_ps();

  if ((reinterpret_cast<uintptr_t>(lhs) & 0x1f) == 0 &&
      (reinterpret_cast<uintptr_t>(rhs) & 0x1f) == 0) {
    for (; lhs != last_aligned; lhs += 16, rhs += 16) {
      ymm_sum_0 = _mm256_fmadd_ps(_mm256_load_ps(lhs + 0),
                                  _mm256_load_ps(rhs + 0), ymm_sum_0);
      ymm_sum_1 = _mm256_fmadd_ps(_mm256_load_ps(lhs + 8),
                                  _mm256_load_ps(rhs + 8), ymm_sum_1);
    }

    if (last >= last_aligned + 8) {
      ymm_sum_0 =
          _mm256_fmadd_ps(_mm256_load_ps(lhs), _mm256_load_ps(rhs), ymm_sum_0);
      lhs += 8;
      rhs += 8;
    }
  } else {
    for (; lhs != last_aligned; lhs += 16, rhs += 16) {
      ymm_sum_0 = _mm256_fmadd_ps(_mm256_loadu_ps(lhs + 0),
                                  _mm256_loadu_ps(rhs + 0), ymm_sum_0);
      ymm_sum_1 = _mm256_fmadd_ps(_mm256_loadu_ps(lhs + 8),
                                  _mm256_loadu_ps(rhs + 8), ymm_sum_1);
    }

    if (last >= last_aligned + 8) {
      ymm_sum_0 = _mm256_fmadd_ps(_mm256_loadu_ps(lhs), _mm256_loadu_ps(rhs),
                                  ymm_sum_0);
      lhs += 8;
      rhs += 8;
    }
  }

  float result = internal::hsum_fp32_v256(_mm256_add_ps(ymm_sum_0, ymm_sum_1));
  for (; lhs != last; ++lhs, ++rhs) {
    result += *lhs * *rhs;
  }

  *distance = -result;
#else
  (void)a;
  (void)b;
  (void)dim;
  (void)distance;
#endif
}

void inner_product_fp32_batch_distance_avx2(const void *const *vectors,
                                            const void *query, size_t n,
                                            size_t dim, float *distances) {
#if defined(__AVX2__)
  for (size_t i = 0; i < n; ++i) {
    inner_product_fp32_distance_avx2(vectors[i], query, dim, &distances[i]);
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
