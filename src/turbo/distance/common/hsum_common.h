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

// Horizontal-sum helpers shared by the fp32 / fp16 distance kernels in the
// avx2 and avx512 ISA directories.  Each helper is guarded by the ISA macro
// active in the including translation unit, so builds without the target ISA
// simply see an empty namespace.

#pragma once

#if defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif

namespace zvec::turbo::internal {

#if defined(__AVX2__) || defined(__AVX512F__)
// Horizontal sum of 8 floats in a __m256 register.
inline float hsum_fp32_v256(__m256 v) {
  // High 128 bits + low 128 bits
  __m128 hi = _mm256_extractf128_ps(v, 1);
  __m128 lo = _mm256_castps256_ps128(v);
  __m128 sum128 = _mm_add_ps(lo, hi);
  // Shuffle and add: [a+b, c+d, a+b, c+d]
  __m128 shuf = _mm_movehdup_ps(sum128);  // [b, b, d, d]
  __m128 sum64 = _mm_add_ps(sum128, shuf);
  // Final: [a+b+c+d, ..., ...]
  __m128 shuf32 = _mm_movehl_ps(sum64, sum64);
  return _mm_cvtss_f32(_mm_add_ss(sum64, shuf32));
}
#endif

#if defined(__AVX512F__)
// Horizontal sum of 16 floats in a __m512 register.
inline float hsum_fp32_v512(__m512 v) {
  __m256 lo = _mm512_castps512_ps256(v);
  __m256 hi = _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(v), 1));
  return hsum_fp32_v256(_mm256_add_ps(lo, hi));
}
#endif

}  // namespace zvec::turbo::internal
