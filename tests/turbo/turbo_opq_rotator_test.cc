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

#include <cmath>
#include <cstring>
#include <random>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "preprocessor/opq_rotator/opq_rotator.h"

using namespace zvec::turbo;

namespace {

// Fill a buffer with random floats.
void fill_random(float *data, size_t count, std::mt19937 &gen) {
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (size_t i = 0; i < count; ++i) data[i] = dist(gen);
}

// Check R^T R == I by rotating the unit basis: the rotated basis vectors must
// stay orthonormal, which is exactly what R^T R == I states.
void expect_orthogonal(const OpqRotator &rot, float tol = 1e-4f) {
  const int dim = rot.in_dim();
  std::vector<std::vector<float>> columns(dim, std::vector<float>(dim));
  std::vector<float> basis(dim, 0.0f);
  for (int i = 0; i < dim; ++i) {
    std::fill(basis.begin(), basis.end(), 0.0f);
    basis[i] = 1.0f;
    rot.apply(basis.data(), columns[i].data());
  }
  for (int i = 0; i < dim; ++i) {
    for (int j = 0; j < dim; ++j) {
      float dot = 0.0f;
      for (int k = 0; k < dim; ++k) dot += columns[i][k] * columns[j][k];
      EXPECT_NEAR(dot, (i == j) ? 1.0f : 0.0f, tol)
          << "R^T R mismatch at (" << i << ", " << j << ")";
    }
  }
}

// Build a reference orthogonal matrix (row-major dim x dim) from a sequence of
// Givens rotations, so tests can compare fit() against a known solution.
std::vector<float> reference_orthogonal(int dim, uint32_t seed) {
  std::vector<float> m(static_cast<size_t>(dim) * dim, 0.0f);
  for (int i = 0; i < dim; ++i) m[i * dim + i] = 1.0f;

  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> angle(-3.14f, 3.14f);
  for (int i = 0; i + 1 < dim; ++i) {
    const float a = angle(gen);
    const float c = std::cos(a);
    const float s = std::sin(a);
    // Rotate rows i and i+1 in place: still orthogonal afterwards.
    for (int k = 0; k < dim; ++k) {
      const float top = m[i * dim + k];
      const float bottom = m[(i + 1) * dim + k];
      m[i * dim + k] = c * top - s * bottom;
      m[(i + 1) * dim + k] = s * top + c * bottom;
    }
  }
  return m;
}

// out = matrix * in, matrix row-major dim x dim.
void matvec(const std::vector<float> &matrix, const float *in, float *out,
            int dim) {
  for (int i = 0; i < dim; ++i) {
    float sum = 0.0f;
    for (int j = 0; j < dim; ++j) sum += matrix[i * dim + j] * in[j];
    out[i] = sum;
  }
}

}  // anonymous namespace

// ---------------------------------------------------------------------------
// create() / apply() / apply_inverse()
// ---------------------------------------------------------------------------

TEST(OpqRotator, CreateProducesUsableRotation) {
  for (int dim : {1, 2, 7, 16, 33, 64}) {
    auto rot = OpqRotator::create(dim);
    ASSERT_TRUE(rot) << "create failed for dim=" << dim;
    EXPECT_EQ(dim, rot->in_dim());
    EXPECT_EQ(dim, rot->out_dim());
    EXPECT_EQ(RotateType::kOpq, rot->rotate_type());
    expect_orthogonal(*rot);
  }

  EXPECT_EQ(nullptr, OpqRotator::create(0));
  EXPECT_EQ(nullptr, OpqRotator::create(-4));
}

TEST(OpqRotator, CreateIsReproducibleWithSeed) {
  const int dim = 24;
  auto a = OpqRotator::create(dim, 7);
  auto b = OpqRotator::create(dim, 7);
  auto c = OpqRotator::create(dim, 8);
  ASSERT_TRUE(a && b && c);

  std::mt19937 gen(1);
  std::vector<float> input(dim);
  fill_random(input.data(), dim, gen);

  std::vector<float> ra(dim), rb(dim), rc(dim);
  a->apply(input.data(), ra.data());
  b->apply(input.data(), rb.data());
  c->apply(input.data(), rc.data());

  for (int i = 0; i < dim; ++i) EXPECT_FLOAT_EQ(ra[i], rb[i]);
  // A different seed must yield a different rotation.
  float diff = 0.0f;
  for (int i = 0; i < dim; ++i) diff += std::fabs(ra[i] - rc[i]);
  EXPECT_GT(diff, 1e-3f);
}

TEST(OpqRotator, RoundTripAndNormPreserved) {
  std::mt19937 gen(42);
  for (int dim : {2, 8, 33, 96}) {
    auto rot = OpqRotator::create(dim);
    ASSERT_TRUE(rot);

    std::vector<float> input(dim);
    fill_random(input.data(), dim, gen);
    std::vector<float> rotated(dim), recovered(dim);
    rot->apply(input.data(), rotated.data());
    rot->apply_inverse(rotated.data(), recovered.data());

    float norm_in = 0.0f, norm_rot = 0.0f;
    for (int i = 0; i < dim; ++i) {
      EXPECT_NEAR(input[i], recovered[i], 1e-4f) << "i=" << i;
      norm_in += input[i] * input[i];
      norm_rot += rotated[i] * rotated[i];
    }
    // Orthogonal transforms preserve the L2 norm.
    EXPECT_NEAR(norm_in, norm_rot, 1e-3f);
  }
}

// ---------------------------------------------------------------------------
// fit(): OPQ step 1, orthogonal Procrustes
// ---------------------------------------------------------------------------

TEST(OpqRotator, FitRecoversKnownRotation) {
  const int dim = 16;
  const size_t num = 512;
  const std::vector<float> r0 = reference_orthogonal(dim, 5);

  std::mt19937 gen(11);
  std::vector<float> x(num * dim);
  fill_random(x.data(), x.size(), gen);

  // x_hat = R0 * x, so the Procrustes solution must be exactly R0.
  std::vector<float> x_hat(num * dim);
  for (size_t i = 0; i < num; ++i) {
    matvec(r0, x.data() + i * dim, x_hat.data() + i * dim, dim);
  }

  auto rot = OpqRotator::create(dim);
  ASSERT_TRUE(rot);
  ASSERT_EQ(0, rot->fit(x.data(), x_hat.data(), num));
  expect_orthogonal(*rot);

  // Compare the fitted rotation against R0 on fresh vectors.
  std::vector<float> probe(dim), expected(dim), actual(dim);
  for (int trial = 0; trial < 5; ++trial) {
    fill_random(probe.data(), dim, gen);
    matvec(r0, probe.data(), expected.data(), dim);
    rot->apply(probe.data(), actual.data());
    for (int i = 0; i < dim; ++i) {
      EXPECT_NEAR(expected[i], actual[i], 1e-4f) << "trial=" << trial;
    }
  }
}

TEST(OpqRotator, FitRejectsInvalidArguments) {
  const int dim = 8;
  const size_t num = 32;
  auto rot = OpqRotator::create(dim);
  ASSERT_TRUE(rot);

  std::mt19937 gen(3);
  std::vector<float> x(num * dim), x_hat(num * dim);
  fill_random(x.data(), x.size(), gen);
  fill_random(x_hat.data(), x_hat.size(), gen);

  // Snapshot the current rotation to prove failures leave it untouched.
  std::vector<float> probe(dim), before(dim), after(dim);
  fill_random(probe.data(), dim, gen);
  rot->apply(probe.data(), before.data());

  EXPECT_NE(0, rot->fit(nullptr, x_hat.data(), num));
  EXPECT_NE(0, rot->fit(x.data(), nullptr, num));
  EXPECT_NE(0, rot->fit(x.data(), x_hat.data(), 0));

  rot->apply(probe.data(), after.data());
  for (int i = 0; i < dim; ++i) EXPECT_FLOAT_EQ(before[i], after[i]);
  expect_orthogonal(*rot);
}

TEST(OpqRotator, TrainIsNoOp) {
  const int dim = 12;
  auto rot = OpqRotator::create(dim);
  ASSERT_TRUE(rot);

  std::mt19937 gen(9);
  std::vector<float> data(dim * 4);
  fill_random(data.data(), data.size(), gen);

  std::vector<float> probe(dim), before(dim), after(dim);
  fill_random(probe.data(), dim, gen);
  rot->apply(probe.data(), before.data());

  rot->train(data.data(), 4, dim * sizeof(float));

  rot->apply(probe.data(), after.data());
  for (int i = 0; i < dim; ++i) EXPECT_FLOAT_EQ(before[i], after[i]);
}

// ---------------------------------------------------------------------------
// Alternating optimization: the caller (a PQ quantizer in production, a
// sub-space mean "codebook" here) drives the loop and OPQ must not increase
// the reconstruction error.
// ---------------------------------------------------------------------------

TEST(OpqRotator, AlternatingLoopReducesReconstructionError) {
  const int dim = 16;
  const int num_chunk = 4;
  const int sub_dim = dim / num_chunk;
  const size_t num = 400;

  // Anisotropic data: the variance decays sharply across dimensions, which is
  // exactly the case OPQ is meant to rebalance across sub-spaces.
  std::mt19937 gen(2024);
  std::normal_distribution<float> normal(0.0f, 1.0f);
  std::vector<float> x(num * dim);
  for (size_t i = 0; i < num; ++i) {
    for (int j = 0; j < dim; ++j) {
      x[i * dim + j] = normal(gen) * std::pow(0.6f, static_cast<float>(j));
    }
  }

  auto rot = OpqRotator::create(dim, 17);
  ASSERT_TRUE(rot);

  std::vector<float> rotated(num * dim);
  std::vector<float> x_hat(num * dim);

  // Stand-in for OPQ step 2: one centroid per sub-space (its mean), which is
  // enough to exercise the alternating loop without pulling in k-means.
  auto reconstruct = [&]() {
    std::vector<float> mean(static_cast<size_t>(num_chunk) * sub_dim, 0.0f);
    for (size_t i = 0; i < num; ++i) {
      for (int j = 0; j < dim; ++j) mean[j] += rotated[i * dim + j];
    }
    for (float &v : mean) v /= static_cast<float>(num);

    double err = 0.0;
    for (size_t i = 0; i < num; ++i) {
      for (int j = 0; j < dim; ++j) {
        x_hat[i * dim + j] = mean[j];
        const float d = rotated[i * dim + j] - mean[j];
        err += static_cast<double>(d) * d;
      }
    }
    return static_cast<float>(err / static_cast<double>(num));
  };

  float first_mse = 0.0f;
  float last_mse = 0.0f;
  for (int it = 0; it < 5; ++it) {
    for (size_t i = 0; i < num; ++i) {
      rot->apply(x.data() + i * dim, rotated.data() + i * dim);
    }
    const float mse = reconstruct();
    if (it == 0) {
      first_mse = mse;
    } else {
      // The Procrustes step is a minimizer, so the error must not grow.
      EXPECT_LE(mse, last_mse * 1.001f + 1e-6f) << "iteration " << it;
    }
    last_mse = mse;
    ASSERT_EQ(0, rot->fit(x.data(), x_hat.data(), num));
  }
  EXPECT_LE(last_mse, first_mse * 1.001f + 1e-6f);
  expect_orthogonal(*rot);
}

// ---------------------------------------------------------------------------
// Serialization
// ---------------------------------------------------------------------------

TEST(OpqRotator, SerializeRoundTrip) {
  const int dim = 20;
  const size_t num = 128;

  std::mt19937 gen(77);
  std::vector<float> x(num * dim), x_hat(num * dim);
  fill_random(x.data(), x.size(), gen);
  fill_random(x_hat.data(), x_hat.size(), gen);

  auto rot = OpqRotator::create(dim);
  ASSERT_TRUE(rot);
  ASSERT_EQ(0, rot->fit(x.data(), x_hat.data(), num));

  std::string blob;
  ASSERT_EQ(0, rot->serialize(&blob));
  EXPECT_EQ(
      sizeof(RotatorSerHeader) + static_cast<size_t>(dim) * dim * sizeof(float),
      blob.size());

  auto restored = OpqRotator::from_blob(blob.data(), blob.size());
  ASSERT_TRUE(restored);
  EXPECT_EQ(dim, restored->in_dim());
  EXPECT_EQ(dim, restored->out_dim());

  std::vector<float> probe(dim), expected(dim), actual(dim);
  for (int trial = 0; trial < 5; ++trial) {
    fill_random(probe.data(), dim, gen);
    rot->apply(probe.data(), expected.data());
    restored->apply(probe.data(), actual.data());
    for (int i = 0; i < dim; ++i) EXPECT_FLOAT_EQ(expected[i], actual[i]);
  }
}

TEST(OpqRotator, FromBlobRejectsMalformedInput) {
  const int dim = 8;
  auto rot = OpqRotator::create(dim);
  ASSERT_TRUE(rot);
  std::string blob;
  ASSERT_EQ(0, rot->serialize(&blob));

  EXPECT_EQ(nullptr, OpqRotator::from_blob(nullptr, blob.size()));
  EXPECT_EQ(nullptr, OpqRotator::from_blob(blob.data(), 0));
  // Truncated payload: the matrix must be present in full.
  EXPECT_EQ(nullptr, OpqRotator::from_blob(blob.data(), blob.size() - 4));

  // Wrong magic.
  std::string bad_magic = blob;
  bad_magic[0] = static_cast<char>(bad_magic[0] + 1);
  EXPECT_EQ(nullptr, OpqRotator::from_blob(bad_magic.data(), bad_magic.size()));

  // Wrong rotator type (kFht blob must not load as OPQ).
  std::string bad_type = blob;
  RotatorSerHeader hdr;
  std::memcpy(&hdr, bad_type.data(), sizeof(hdr));
  hdr.rotator_type = static_cast<uint16_t>(RotateType::kFht);
  std::memcpy(&bad_type[0], &hdr, sizeof(hdr));
  EXPECT_EQ(nullptr, OpqRotator::from_blob(bad_type.data(), bad_type.size()));

  // Inconsistent payload size for the declared dimension.
  std::string bad_size = blob;
  std::memcpy(&hdr, bad_size.data(), sizeof(hdr));
  hdr.payload_size = static_cast<uint32_t>(hdr.payload_size - sizeof(float));
  std::memcpy(&bad_size[0], &hdr, sizeof(hdr));
  EXPECT_EQ(nullptr, OpqRotator::from_blob(bad_size.data(), bad_size.size()));
}
