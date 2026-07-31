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
// Kernel ported from ailego/math/inner_product_matrix_fp16_neon.cc:
// halves are widened to fp32 with vcvt_f32_f16 and accumulated with dual
// low/high accumulators; the tail is handled through a zero-padded staging
// buffer (zero lanes contribute nothing to the dot product).

#include "neon/fp16/inner_product.h"
#if defined(__ARM_NEON) && defined(__aarch64__)
#include <arm_neon.h>
#endif
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace zvec::turbo::neon {

void inner_product_fp16_distance_neon(const void *a, const void *b, size_t dim,
                                      float *distance) {
#if defined(__ARM_NEON) && defined(__aarch64__)
  const uint16_t *lhs = reinterpret_cast<const uint16_t *>(a);
  const uint16_t *rhs = reinterpret_cast<const uint16_t *>(b);
  const uint16_t *last = lhs + dim;
  const uint16_t *last_aligned = lhs + ((dim >> 3) << 3);

  float32x4_t v_sum_0 = vdupq_n_f32(0);
  float32x4_t v_sum_1 = vdupq_n_f32(0);

  for (; lhs != last_aligned; lhs += 8, rhs += 8) {
    float16x8_t v_m = vld1q_f16(reinterpret_cast<const float16_t *>(lhs));
    float16x8_t v_q = vld1q_f16(reinterpret_cast<const float16_t *>(rhs));
    v_sum_0 = vfmaq_f32(v_sum_0, vcvt_f32_f16(vget_low_f16(v_m)),
                        vcvt_f32_f16(vget_low_f16(v_q)));
    v_sum_1 = vfmaq_f32(v_sum_1, vcvt_f32_f16(vget_high_f16(v_m)),
                        vcvt_f32_f16(vget_high_f16(v_q)));
  }

  if (lhs != last) {
    // Zero-padded staging buffers: padded lanes yield product 0.
    uint16_t buf_m[8] = {0};
    uint16_t buf_q[8] = {0};
    size_t remain = static_cast<size_t>(last - lhs);
    memcpy(buf_m, lhs, remain * sizeof(uint16_t));
    memcpy(buf_q, rhs, remain * sizeof(uint16_t));
    float16x8_t v_m = vld1q_f16(reinterpret_cast<const float16_t *>(buf_m));
    float16x8_t v_q = vld1q_f16(reinterpret_cast<const float16_t *>(buf_q));
    v_sum_0 = vfmaq_f32(v_sum_0, vcvt_f32_f16(vget_low_f16(v_m)),
                        vcvt_f32_f16(vget_low_f16(v_q)));
    v_sum_1 = vfmaq_f32(v_sum_1, vcvt_f32_f16(vget_high_f16(v_m)),
                        vcvt_f32_f16(vget_high_f16(v_q)));
  }

  *distance = -vaddvq_f32(vaddq_f32(v_sum_0, v_sum_1));
#else
  (void)a;
  (void)b;
  (void)dim;
  (void)distance;
#endif
}

void inner_product_fp16_batch_distance_neon(const void *const *vectors,
                                            const void *query, size_t n,
                                            size_t dim, float *distances) {
#if defined(__ARM_NEON) && defined(__aarch64__)
  for (size_t i = 0; i < n; ++i) {
    inner_product_fp16_distance_neon(vectors[i], query, dim, &distances[i]);
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
