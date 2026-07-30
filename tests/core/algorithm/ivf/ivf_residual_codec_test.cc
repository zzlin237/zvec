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

#include "ivf_residual_codec.h"
#include <cmath>
#include <cstring>
#include <random>
#include <vector>
#include <gtest/gtest.h>
#include <zvec/ailego/container/params.h>

using namespace zvec;
using namespace zvec::core;
using zvec::ailego::NumericalVector;

static IndexMeta make_meta(size_t dim, const std::string &metric) {
  IndexMeta meta;
  meta.set_meta(IndexMeta::DataType::DT_FP32, dim);
  meta.set_metric(metric, 0, ailego::Params());
  return meta;
}

static std::shared_ptr<MultiPassIndexHolder<IndexMeta::DataType::DT_FP32>>
make_random_holder(size_t count, size_t dim, uint32_t seed = 42) {
  auto holder =
      std::make_shared<MultiPassIndexHolder<IndexMeta::DataType::DT_FP32>>(dim);
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (size_t i = 0; i < count; ++i) {
    NumericalVector<float> vec(dim);
    for (size_t j = 0; j < dim; ++j) vec[j] = dist(gen);
    holder->emplace(i + 1, vec);
  }
  return holder;
}

TEST(IVFResidualCodecTest, SupportsMetric) {
  EXPECT_TRUE(IVFResidualCodec::SupportsMetric("SquaredEuclidean"));
  EXPECT_TRUE(IVFResidualCodec::SupportsMetric("Cosine"));
  EXPECT_TRUE(IVFResidualCodec::SupportsMetric("InnerProduct"));
  EXPECT_FALSE(IVFResidualCodec::SupportsMetric("MipsSquaredEuclidean"));
  EXPECT_FALSE(IVFResidualCodec::SupportsMetric(""));
}

TEST(IVFResidualCodecTest, InitDefaultsAndValidation) {
  //! Defaults: quantizer_class=PqInt8Quantizer, num_chunk=8
  IVFResidualCodec codec;
  ASSERT_EQ(codec.init(make_meta(32, "SquaredEuclidean"), ailego::Params()),
            0);
  EXPECT_EQ(codec.num_chunk(), 8u);
  EXPECT_EQ(codec.dim(), 32u);
  EXPECT_EQ(codec.quantizer_class(), "PqInt8Quantizer");
  EXPECT_EQ(codec.code_size(), 8u);  // uint8[num_chunk]

  //! dim not divisible by num_chunk
  IVFResidualCodec bad;
  ailego::Params params;
  params.set("num_chunk", 7);
  EXPECT_NE(bad.init(make_meta(32, "SquaredEuclidean"), params), 0);

  //! unsupported metric
  IVFResidualCodec mips;
  EXPECT_NE(mips.init(make_meta(32, "MipsSquaredEuclidean"), ailego::Params()),
            0);

  //! "dim" param overrides meta dimension (load side)
  IVFResidualCodec dimmed;
  ailego::Params dim_params;
  dim_params.set("dim", 16u);
  ASSERT_EQ(dimmed.init(make_meta(32, "SquaredEuclidean"), dim_params), 0);
  EXPECT_EQ(dimmed.dim(), 16u);
}

TEST(IVFResidualCodecTest, CosineCentroidNormalization) {
  const size_t dim = 8;
  IVFResidualCodec codec;
  ASSERT_EQ(codec.init(make_meta(dim, "Cosine"), ailego::Params()), 0);

  std::vector<float> centroids(dim, 0.0f);
  centroids[0] = 3.0f;
  centroids[1] = 4.0f;
  ASSERT_EQ(codec.set_centroids(centroids, 1), 0);
  EXPECT_NEAR(codec.centroid(0)[0], 0.6f, 1e-6f);
  EXPECT_NEAR(codec.centroid(0)[1], 0.8f, 1e-6f);

  //! size mismatch is rejected
  IVFResidualCodec other;
  ASSERT_EQ(other.init(make_meta(dim, "Cosine"), ailego::Params()), 0);
  EXPECT_NE(other.set_centroids(std::vector<float>(dim - 1), 1), 0);
}

TEST(IVFResidualCodecTest, ComputeResidual) {
  const size_t dim = 8;

  //! L2: plain subtraction
  IVFResidualCodec l2;
  ASSERT_EQ(l2.init(make_meta(dim, "SquaredEuclidean"), ailego::Params()), 0);
  std::vector<float> vec(dim, 2.0f), centroid(dim, 0.5f), out(dim);
  l2.compute_residual(vec.data(), centroid.data(), out.data());
  for (size_t d = 0; d < dim; ++d) {
    EXPECT_NEAR(out[d], 1.5f, 1e-6f);
  }

  //! Cosine: normalize BEFORE subtracting the centroid
  IVFResidualCodec cos;
  ASSERT_EQ(cos.init(make_meta(dim, "Cosine"), ailego::Params()), 0);
  std::vector<float> cvec(dim, 0.0f), ccent(dim, 0.0f);
  cvec[1] = 5.0f;   // unit(cvec) = e1
  ccent[1] = 1.0f;  // residual = e1 - e1 = 0
  cos.compute_residual(cvec.data(), ccent.data(), out.data());
  for (size_t d = 0; d < dim; ++d) {
    EXPECT_NEAR(out[d], 0.0f, 1e-6f);
  }
}

TEST(IVFResidualCodecTest, BuildListQueryDis0Semantics) {
  const size_t dim = 32;
  const size_t nlist = 2;
  std::mt19937 gen(7);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> centroids(nlist * dim);
  for (auto &v : centroids) v = dist(gen);
  std::vector<float> query(dim);
  for (auto &v : query) v = dist(gen);

  //! IP: LUT on the raw query, dis0 = -<q, c>
  IVFResidualCodec ip;
  ASSERT_EQ(ip.init(make_meta(dim, "InnerProduct"), ailego::Params()), 0);
  ASSERT_EQ(ip.set_centroids(centroids, nlist), 0);
  ASSERT_EQ(ip.train(make_random_holder(512, dim)), 0);
  std::vector<float> lut(ip.lut_size() / sizeof(float));
  float dis0 = 0.0f;
  ip.build_list_query(query.data(), 1, lut.data(), &dis0);
  double dot = 0.0;
  for (size_t d = 0; d < dim; ++d) {
    dot += static_cast<double>(query[d]) * centroids[dim + d];
  }
  EXPECT_NEAR(dis0, static_cast<float>(-dot), 1e-4f);

  //! L2: LUT on the residual, dis0 = 0
  IVFResidualCodec l2;
  ASSERT_EQ(l2.init(make_meta(dim, "SquaredEuclidean"), ailego::Params()), 0);
  ASSERT_EQ(l2.set_centroids(centroids, nlist), 0);
  ASSERT_EQ(l2.train(make_random_holder(512, dim)), 0);
  dis0 = -1.0f;
  std::vector<float> l2_lut(l2.lut_size() / sizeof(float));
  l2.build_list_query(query.data(), 0, l2_lut.data(), &dis0);
  EXPECT_EQ(dis0, 0.0f);
}

TEST(IVFResidualCodecTest, SerializeDeserializeRoundtrip) {
  const size_t dim = 32;
  const size_t nlist = 4;
  std::mt19937 gen(11);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> centroids(nlist * dim);
  for (auto &v : centroids) v = dist(gen);

  IVFResidualCodec codec;
  ASSERT_EQ(codec.init(make_meta(dim, "SquaredEuclidean"), ailego::Params()),
            0);
  ASSERT_EQ(codec.set_centroids(centroids, nlist), 0);
  ASSERT_EQ(codec.train(make_random_holder(512, dim)), 0);

  std::string blob;
  ASSERT_EQ(codec.serialize_codebook(&blob), 0);
  ASSERT_FALSE(blob.empty());

  //! Restore into a second codec, mirroring the load path (no use_zero_mean)
  IVFResidualCodec loaded;
  ailego::Params load_params;
  load_params.set("dim", static_cast<uint32_t>(dim));
  ASSERT_EQ(loaded.init(make_meta(dim, "SquaredEuclidean"), load_params), 0);
  ASSERT_EQ(loaded.set_centroids(centroids, nlist), 0);
  ASSERT_EQ(loaded.deserialize_codebook(blob.data(), blob.size()), 0);

  //! Same encoding and same ADC distances on both sides
  std::vector<float> vec(dim);
  for (auto &v : vec) v = dist(gen);
  std::vector<uint8_t> code_a(codec.code_size());
  std::vector<uint8_t> code_b(loaded.code_size());
  codec.encode(vec.data(), 2, code_a.data());
  loaded.encode(vec.data(), 2, code_b.data());
  ASSERT_EQ(code_a.size(), code_b.size());
  EXPECT_EQ(std::memcmp(code_a.data(), code_b.data(), code_a.size()), 0);

  std::vector<float> query(dim);
  for (auto &v : query) v = dist(gen);
  std::vector<float> lut_a(codec.lut_size() / sizeof(float));
  std::vector<float> lut_b(loaded.lut_size() / sizeof(float));
  float dis0_a = 0.0f, dis0_b = 0.0f;
  codec.build_list_query(query.data(), 2, lut_a.data(), &dis0_a);
  loaded.build_list_query(query.data(), 2, lut_b.data(), &dis0_b);
  EXPECT_EQ(dis0_a, dis0_b);

  float dist_a = 0.0f, dist_b = 0.0f;
  codec.batch_distance(code_a.data(), 1, codec.code_size(), lut_a.data(),
                       &dist_a);
  loaded.batch_distance(code_b.data(), 1, loaded.code_size(), lut_b.data(),
                        &dist_b);
  EXPECT_FLOAT_EQ(dist_a, dist_b);

  //! ADC distance approximates the true residual distance
  std::vector<float> resid_q(dim), resid_v(dim);
  codec.compute_residual(query.data(), codec.centroid(2), resid_q.data());
  codec.compute_residual(vec.data(), codec.centroid(2), resid_v.data());
  float ref = 0.0f;
  for (size_t d = 0; d < dim; ++d) {
    float diff = resid_q[d] - resid_v[d];
    ref += diff * diff;
  }
  EXPECT_NEAR(dist_a, ref, 0.5f * ref + 0.5f);
}
