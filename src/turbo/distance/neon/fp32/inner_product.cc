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

// NEON is enabled by default on aarch64, so no per-file march flag is
// needed. On non-ARM builds each function falls back to a no-op stub
// guarded by #if defined(__ARM_NEON) && defined(__aarch64__).
//
// Kernel ported from ailego/math/inner_product_matrix_fp32_neon.cc:
// 2x-unrolled dual accumulators and a scalar tail, adapted to the turbo
// DistanceFunc signature (negated dot).

#include "neon/fp32/inner_product.h"
#if defined(__ARM_NEON) && defined(__aarch64__)
#include <arm_neon.h>
#endif
#include <cstddef>

namespace zvec::turbo::neon {

void inner_product_fp32_distance_neon(const void *a, const void *b, size_t dim,
                                      float *distance) {
#if defined(__ARM_NEON) && defined(__aarch64__)
  const float *lhs = reinterpret_cast<const float *>(a);
  const float *rhs = reinterpret_cast<const float *>(b);
  const float *last = lhs + dim;
  const float *last_aligned = lhs + ((dim >> 3) << 3);

  float32x4_t v_sum_0 = vdupq_n_f32(0);
  float32x4_t v_sum_1 = vdupq_n_f32(0);

  for (; lhs != last_aligned; lhs += 8, rhs += 8) {
    v_sum_0 = vfmaq_f32(v_sum_0, vld1q_f32(lhs + 0), vld1q_f32(rhs + 0));
    v_sum_1 = vfmaq_f32(v_sum_1, vld1q_f32(lhs + 4), vld1q_f32(rhs + 4));
  }
  if (last >= last_aligned + 4) {
    v_sum_0 = vfmaq_f32(v_sum_0, vld1q_f32(lhs), vld1q_f32(rhs));
    lhs += 4;
    rhs += 4;
  }

  float result = vaddvq_f32(vaddq_f32(v_sum_0, v_sum_1));
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

void inner_product_fp32_batch_distance_neon(const void *const *vectors,
                                            const void *query, size_t n,
                                            size_t dim, float *distances) {
#if defined(__ARM_NEON) && defined(__aarch64__)
  for (size_t i = 0; i < n; ++i) {
    inner_product_fp32_distance_neon(vectors[i], query, dim, &distances[i]);
  }
#else
  (void)vectors;
  (void)query;
  (void)n;
  (void)dim;
  (void)distances;
#endif
}

}  // namespace zvec::turbo::neon
