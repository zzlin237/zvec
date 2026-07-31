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

// Cross-ISA consistency tests for the fp32 / fp16 distance kernels behind
// get_distance_func / get_batch_distance_func: every SIMD variant must
// agree with the scalar baseline within accumulation-order tolerance.

#include <cmath>
#include <random>
#include <vector>
#include <gtest/gtest.h>
#include <zvec/ailego/utility/float_helper.h>
#include <zvec/turbo/turbo.h>

using namespace zvec::turbo;
using zvec::ailego::Float16;

namespace {

// Cover multiples and non-multiples of every SIMD width (4/8/16 lanes plus
// the 2x-unrolled main loops) to exercise main-loop, half-step and tail.
const size_t kDims[] = {3, 7, 16, 17, 32, 33, 64, 100, 130};

const MetricType kMetrics[] = {MetricType::kSquaredEuclidean,
                               MetricType::kInnerProduct, MetricType::kCosine};

// SIMD arches under test. Unsupported arches fall back to the scalar kernel
// inside get_distance_func, so the comparison below stays valid everywhere.
const CpuArchType kArches[] = {CpuArchType::kAVX512, CpuArchType::kAVX2,
                               CpuArchType::kNEON, CpuArchType::kAuto};

std::vector<float> RandomVector(size_t dim, std::mt19937 &gen) {
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> vec(dim);
  for (auto &v : vec) {
    v = dist(gen);
  }
  return vec;
}

// Compare one SIMD arch against the scalar baseline for a single metric and
// data type, over single and batch entry points, aligned and offset inputs.
template <typename T>
void CheckDistanceConsistency(MetricType metric, DataType data_type,
                              float tolerance) {
  std::mt19937 gen(20260731);
  const size_t kBatch = 5;

  DistanceFunc scalar_fn = get_distance_func(
      metric, data_type, QuantizeType::kDefault, CpuArchType::kScalar);
  BatchDistanceFunc scalar_batch_fn = get_batch_distance_func(
      metric, data_type, QuantizeType::kDefault, CpuArchType::kScalar);
  ASSERT_TRUE(scalar_fn);
  ASSERT_TRUE(scalar_batch_fn);

  for (size_t dim : kDims) {
    // One extra leading element so that data() + 1 gives a misaligned view.
    std::vector<float> query_f = RandomVector(dim + 1, gen);
    std::vector<std::vector<float>> records_f;
    for (size_t i = 0; i < kBatch; ++i) {
      records_f.push_back(RandomVector(dim + 1, gen));
    }

    std::vector<T> query(query_f.begin(), query_f.end());
    std::vector<std::vector<T>> records;
    for (const auto &r : records_f) {
      records.emplace_back(r.begin(), r.end());
    }

    for (CpuArchType arch : kArches) {
      DistanceFunc fn =
          get_distance_func(metric, data_type, QuantizeType::kDefault, arch);
      BatchDistanceFunc batch_fn = get_batch_distance_func(
          metric, data_type, QuantizeType::kDefault, arch);
      ASSERT_TRUE(fn);
      ASSERT_TRUE(batch_fn);

      for (size_t offset = 0; offset <= 1; ++offset) {
        const void *q = query.data() + offset;
        std::vector<const void *> vecs;
        for (const auto &r : records) {
          vecs.push_back(r.data() + offset);
        }

        for (size_t i = 0; i < kBatch; ++i) {
          float expected = 0.0f, actual = 0.0f;
          scalar_fn(vecs[i], q, dim, &expected);
          fn(vecs[i], q, dim, &actual);
          EXPECT_NEAR(expected, actual,
                      tolerance * (1.0f + std::fabs(expected)))
              << "single metric=" << static_cast<int>(metric)
              << " arch=" << static_cast<int>(arch) << " dim=" << dim
              << " offset=" << offset << " record=" << i;
        }

        std::vector<float> expected(kBatch, 0.0f), actual(kBatch, 0.0f);
        scalar_batch_fn(vecs.data(), q, kBatch, dim, expected.data());
        batch_fn(vecs.data(), q, kBatch, dim, actual.data());
        for (size_t i = 0; i < kBatch; ++i) {
          EXPECT_NEAR(expected[i], actual[i],
                      tolerance * (1.0f + std::fabs(expected[i])))
              << "batch metric=" << static_cast<int>(metric)
              << " arch=" << static_cast<int>(arch) << " dim=" << dim
              << " offset=" << offset << " record=" << i;
        }
      }
    }
  }
}

}  // namespace

TEST(TurboDistance, Fp32SimdMatchesScalar) {
  for (MetricType metric : kMetrics) {
    CheckDistanceConsistency<float>(metric, DataType::kFp32, 1e-4f);
  }
}

TEST(TurboDistance, Fp16SimdMatchesScalar) {
  for (MetricType metric : kMetrics) {
    CheckDistanceConsistency<Float16>(metric, DataType::kFp16, 1e-3f);
  }
}

// The auto-dispatched kernel must expose the exact same -dot / 1 + ip / SSD
// semantics as the scalar reference implementation.
TEST(TurboDistance, SemanticsAgainstReference) {
  std::mt19937 gen(15583);
  const size_t dim = 100;
  std::vector<float> a = RandomVector(dim, gen);
  std::vector<float> b = RandomVector(dim, gen);

  float dot = 0.0f, ssd = 0.0f;
  for (size_t i = 0; i < dim; ++i) {
    dot += a[i] * b[i];
    ssd += (a[i] - b[i]) * (a[i] - b[i]);
  }

  float out = 0.0f;
  get_distance_func(MetricType::kInnerProduct, DataType::kFp32,
                    QuantizeType::kDefault)(a.data(), b.data(), dim, &out);
  EXPECT_NEAR(-dot, out, 1e-4f * (1.0f + std::fabs(dot)));

  get_distance_func(MetricType::kCosine, DataType::kFp32,
                    QuantizeType::kDefault)(a.data(), b.data(), dim, &out);
  EXPECT_NEAR(1.0f - dot, out, 1e-4f * (1.0f + std::fabs(dot)));

  get_distance_func(MetricType::kSquaredEuclidean, DataType::kFp32,
                    QuantizeType::kDefault)(a.data(), b.data(), dim, &out);
  EXPECT_NEAR(ssd, out, 1e-4f * (1.0f + std::fabs(ssd)));
}

// Unsupported combinations must keep returning nullptr after the SIMD
// dispatch expansion.
TEST(TurboDistance, UnsupportedCombinationsReturnNull) {
  EXPECT_FALSE(get_distance_func(MetricType::kSquaredEuclidean, DataType::kFp32,
                                 QuantizeType::kUniform));
  EXPECT_FALSE(get_distance_func(MetricType::kSquaredEuclidean, DataType::kFp16,
                                 QuantizeType::kUniform));
  EXPECT_FALSE(get_batch_distance_func(
      MetricType::kSquaredEuclidean, DataType::kFp32, QuantizeType::kUniform));
}
