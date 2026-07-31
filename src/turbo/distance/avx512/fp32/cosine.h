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

#pragma once

#include <cstddef>

namespace zvec::turbo::avx512 {

// Compute cosine distance between a single FP32 vector pair with AVX512.
// Assumes both vectors are normalized; derived as 1 + ip from the negated
// inner product kernel (same contract as the scalar kernel).
void cosine_fp32_distance_avx512(const void *a, const void *b, size_t dim,
                                 float *distance);

// Batch version of cosine_fp32_distance_avx512.
void cosine_fp32_batch_distance_avx512(const void *const *vectors,
                                       const void *query, size_t n, size_t dim,
                                       float *distances);

}  // namespace zvec::turbo::avx512
