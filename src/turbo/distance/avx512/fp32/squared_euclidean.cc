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
// Kernel ported from ailego/math/euclidean_distance_matrix_fp32_avx512.cc:
// 2x-unrolled dual accumulators, aligned/unaligned dual load paths and a
// masked tail, adapted to the turbo DistanceFunc signature.

#include "avx512/fp32/squared_euclidean.h"
#if defined(__AVX512F__)
#include <immintrin.h>
#include "common/hsum_common.h"
#endif
#include <cstddef>
#include <cstdint>

namespace zvec::turbo::avx512 {

void squared_euclidean_fp32_distance_avx512(const void *a, const void *b,
                                            size_t dim, float *distance) {
#if defined(__AVX512F__)
  const float *lhs = reinterpret_cast<const float *>(a);
  const float *rhs = reinterpret_cast<const float *>(b);
  const float *last = lhs + dim;
  const float *last_aligned = lhs + ((dim >> 5) << 5);

  __m512 zmm_sum_0 = _mm512_setzero_ps();
  __m512 zmm_sum_1 = _mm512_setzero_ps();

  if ((reinterpret_cast<uintptr_t>(lhs) & 0x3f) == 0 &&
      (reinterpret_cast<uintptr_t>(rhs) & 0x3f) == 0) {
    for (; lhs != last_aligned; lhs += 32, rhs += 32) {
      __m512 zmm_d_0 =
          _mm512_sub_ps(_mm512_load_ps(lhs + 0), _mm512_load_ps(rhs + 0));
      __m512 zmm_d_1 =
          _mm512_sub_ps(_mm512_load_ps(lhs + 16), _mm512_load_ps(rhs + 16));
      zmm_sum_0 = _mm512_fmadd_ps(zmm_d_0, zmm_d_0, zmm_sum_0);
      zmm_sum_1 = _mm512_fmadd_ps(zmm_d_1, zmm_d_1, zmm_sum_1);
    }

    if (last >= last_aligned + 16) {
      __m512 zmm_d = _mm512_sub_ps(_mm512_load_ps(lhs), _mm512_load_ps(rhs));
      zmm_sum_0 = _mm512_fmadd_ps(zmm_d, zmm_d, zmm_sum_0);
      lhs += 16;
      rhs += 16;
    }
  } else {
    for (; lhs != last_aligned; lhs += 32, rhs += 32) {
      __m512 zmm_d_0 =
          _mm512_sub_ps(_mm512_loadu_ps(lhs + 0), _mm512_loadu_ps(rhs + 0));
      __m512 zmm_d_1 =
          _mm512_sub_ps(_mm512_loadu_ps(lhs + 16), _mm512_loadu_ps(rhs + 16));
      zmm_sum_0 = _mm512_fmadd_ps(zmm_d_0, zmm_d_0, zmm_sum_0);
      zmm_sum_1 = _mm512_fmadd_ps(zmm_d_1, zmm_d_1, zmm_sum_1);
    }

    if (last >= last_aligned + 16) {
      __m512 zmm_d = _mm512_sub_ps(_mm512_loadu_ps(lhs), _mm512_loadu_ps(rhs));
      zmm_sum_0 = _mm512_fmadd_ps(zmm_d, zmm_d, zmm_sum_0);
      lhs += 16;
      rhs += 16;
    }
  }

  zmm_sum_0 = _mm512_add_ps(zmm_sum_0, zmm_sum_1);
  if (lhs != last) {
    __mmask16 mask = static_cast<__mmask16>((1u << (last - lhs)) - 1);
    __m512 zmm_undefined = _mm512_undefined_ps();
    __m512 zmm_d = _mm512_mask_sub_ps(
        zmm_undefined, mask, _mm512_mask_loadu_ps(zmm_undefined, mask, lhs),
        _mm512_mask_loadu_ps(zmm_undefined, mask, rhs));
    zmm_sum_0 = _mm512_mask3_fmadd_ps(zmm_d, zmm_d, zmm_sum_0, mask);
  }

  *distance = internal::hsum_fp32_v512(zmm_sum_0);
#else
  (void)a;
  (void)b;
  (void)dim;
  (void)distance;
#endif
}

void squared_euclidean_fp32_batch_distance_avx512(const void *const *vectors,
                                                  const void *query, size_t n,
                                                  size_t dim,
                                                  float *distances) {
#if defined(__AVX512F__)
  for (size_t i = 0; i < n; ++i) {
    squared_euclidean_fp32_distance_avx512(vectors[i], query, dim,
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
