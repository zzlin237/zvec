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

//! Tests for IVF with a decoupled turbo quantizer (opaque base pointer,
//! raw-vector quantization, independent quantizer segment persistence).

#include <cmath>
#include <random>
#include <set>
#include <vector>
#include <gtest/gtest.h>
#include "zvec/core/framework/index_framework.h"
#include "ivf_builder.h"
#include "ivf_searcher.h"

using namespace zvec::core;
using namespace zvec::ailego;
using namespace std;

class IVFTurboQuantizerTest : public testing::Test {
 protected:
  void SetUp() override;
  void TearDown() override;

  void prepare_holder(uint32_t num);

  //! Build, dump and reload an IVF index with the given builder params.
  //! Returns the loaded searcher/container pair ready for search.
  void build_dump_load(const Params &builder_params, IVFSearcher *searcher,
                       IndexStorage::Pointer *container);

  //! Brute-force ground truth ids for the stored base vectors.
  vector<uint64_t> brute_force_topk(const float *query, size_t topk,
                                    bool cosine) const;

  void recall_at(const Params &builder_params, size_t topk, bool cosine,
                 float *recall);

  uint32_t dimension_{32};
  uint32_t base_num_{1000};
  uint32_t query_num_{32};
  vector<float> base_data_{};
  vector<float> query_data_{};
  string index_path_{};
  IndexHolder::Pointer holder_{};
};

void IVFTurboQuantizerTest::SetUp() {
  index_path_ = "./ivf_turbo_quantizer.index";
  std::mt19937 gen(12345);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  base_data_.resize(base_num_ * dimension_);
  for (auto &v : base_data_) {
    v = dist(gen);
  }
  query_data_.resize(query_num_ * dimension_);
  for (auto &v : query_data_) {
    v = dist(gen);
  }
}

void IVFTurboQuantizerTest::TearDown() {
  File::RemovePath(index_path_);
}

void IVFTurboQuantizerTest::prepare_holder(uint32_t num) {
  auto holder = std::make_shared<MultiPassIndexHolder<
      IndexMeta::DataType::DT_FP32>>(dimension_);
  for (uint32_t i = 0; i < num; ++i) {
    NumericalVector<float> vec(dimension_);
    for (uint32_t j = 0; j < dimension_; ++j) {
      vec[j] = base_data_[i * dimension_ + j];
    }
    holder->emplace(i, vec);
  }
  holder_ = holder;
}

void IVFTurboQuantizerTest::build_dump_load(
    const Params &builder_params, IVFSearcher *searcher,
    IndexStorage::Pointer *container) {
  IndexMeta meta;
  meta.set_meta(IndexMeta::DataType::DT_FP32, dimension_);
  meta.set_metric(builder_params.has(PARAM_IVF_BUILDER_CONVERTER_CLASS)
                      ? "Cosine"
                      : "SquaredEuclidean",
                  0, Params());

  IVFBuilder builder;
  ASSERT_EQ(0, builder.init(meta, builder_params));

  prepare_holder(base_num_);
  ASSERT_EQ(0, builder.train(nullptr, holder_));
  ASSERT_EQ(0, builder.build(nullptr, holder_));

  IndexDumper::Pointer dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_TRUE(!!dumper);
  ASSERT_EQ(0, dumper->create(index_path_));
  ASSERT_EQ(0, builder.dump(dumper));
  EXPECT_EQ((size_t)base_num_, builder.stats().dumped_count());
  ASSERT_EQ(0, dumper->close());

  Params searcher_params;
  searcher_params.set(PARAM_IVF_SEARCHER_SCAN_RATIO, 1.0);
  searcher_params.set(PARAM_IVF_SEARCHER_BRUTE_FORCE_THRESHOLD, 0);
  ASSERT_EQ(0, searcher->init(searcher_params));

  *container = IndexFactory::CreateStorage("MMapFileReadStorage");
  ASSERT_TRUE(!!*container);
  Params container_params;
  container_params.set("proxima.mmap_file.container.memory_warmup", true);
  (*container)->init(container_params);
  ASSERT_EQ(0, (*container)->open(index_path_, false));
  ASSERT_EQ(0, searcher->load(*container, IndexMetric::Pointer()));
}

vector<uint64_t> IVFTurboQuantizerTest::brute_force_topk(const float *query,
                                                         size_t topk,
                                                         bool cosine) const {
  vector<pair<float, uint64_t>> dists(base_num_);
  double qnorm = 0.0;
  for (uint32_t j = 0; j < dimension_; ++j) {
    qnorm += query[j] * query[j];
  }
  for (uint32_t i = 0; i < base_num_; ++i) {
    const float *vec = &base_data_[i * dimension_];
    if (cosine) {
      double dot = 0.0, vnorm = 0.0;
      for (uint32_t j = 0; j < dimension_; ++j) {
        dot += query[j] * vec[j];
        vnorm += vec[j] * vec[j];
      }
      double denom = std::sqrt(qnorm * vnorm);
      dists[i] = {static_cast<float>(denom == 0.0 ? 1.0 : 1.0 - dot / denom),
                  i};
    } else {
      float d = 0.0f;
      for (uint32_t j = 0; j < dimension_; ++j) {
        float diff = query[j] - vec[j];
        d += diff * diff;
      }
      dists[i] = {d, i};
    }
  }
  std::sort(dists.begin(), dists.end());
  vector<uint64_t> gt;
  for (size_t i = 0; i < topk && i < dists.size(); ++i) {
    gt.push_back(dists[i].second);
  }
  return gt;
}

void IVFTurboQuantizerTest::recall_at(const Params &builder_params,
                                      size_t topk, bool cosine,
                                      float *recall) {
  IVFSearcher searcher;
  IndexStorage::Pointer container;
  build_dump_load(builder_params, &searcher, &container);

  auto context = searcher.create_context();
  ASSERT_TRUE(!!context);
  context->set_topk(topk);

  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dimension_);
  size_t hit = 0;
  for (uint32_t q = 0; q < query_num_; ++q) {
    const float *query = &query_data_[q * dimension_];
    ASSERT_EQ(0, searcher.search_impl(query, qmeta, context));
    const IndexDocumentList &result = context->result(0);
    ASSERT_EQ(topk, result.size());
    auto gt = brute_force_topk(query, topk, cosine);
    std::set<uint64_t> gt_set(gt.begin(), gt.end());
    for (size_t i = 0; i < result.size(); ++i) {
      if (gt_set.count(result[i].key())) {
        ++hit;
      }
    }
  }
  *recall = static_cast<float>(hit) / (query_num_ * topk);
}

TEST_F(IVFTurboQuantizerTest, TestL2PqInt8) {
  Params params;
  params.set(PARAM_IVF_BUILDER_CENTROID_COUNT, "16");
  params.set(PARAM_IVF_BUILDER_CLUSTER_CLASS, "KmeansCluster");
  params.set(PARAM_IVF_BUILDER_TURBO_QUANTIZER_CLASS, "PqInt8Quantizer");
  params.set("num_chunk", 8);

  float recall = 0.0f;
  recall_at(params, 10, false, &recall);
  // Loose threshold: raw-vector PQ on random data.
  EXPECT_GT(recall, 0.5f);
}

TEST_F(IVFTurboQuantizerTest, TestCosinePqInt8) {
  Params params;
  params.set(PARAM_IVF_BUILDER_CENTROID_COUNT, "16");
  params.set(PARAM_IVF_BUILDER_CLUSTER_CLASS, "KmeansCluster");
  params.set(PARAM_IVF_BUILDER_CONVERTER_CLASS, "CosineNormalizeConverter");
  params.set(PARAM_IVF_BUILDER_TURBO_QUANTIZER_CLASS, "PqInt8Quantizer");
  params.set("num_chunk", 8);

  float recall = 0.0f;
  recall_at(params, 10, true, &recall);
  EXPECT_GT(recall, 0.5f);
}

TEST_F(IVFTurboQuantizerTest, TestBatchSearchAndQuantizerRestore) {
  Params params;
  params.set(PARAM_IVF_BUILDER_CENTROID_COUNT, "16");
  params.set(PARAM_IVF_BUILDER_CLUSTER_CLASS, "KmeansCluster");
  params.set(PARAM_IVF_BUILDER_TURBO_QUANTIZER_CLASS, "PqInt8Quantizer");
  params.set("num_chunk", 8);

  IVFSearcher searcher;
  IndexStorage::Pointer container;
  build_dump_load(params, &searcher, &container);

  auto context = searcher.create_context();
  ASSERT_TRUE(!!context);
  const size_t topk = 5;
  context->set_topk(topk);

  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dimension_);
  // Batch search: multi-query pointer advancement over the LUT buffer must
  // stay consistent with per-query results.
  ASSERT_EQ(0, searcher.search_impl(query_data_.data(), qmeta, query_num_,
                                    context));
  for (uint32_t q = 0; q < query_num_; ++q) {
    const IndexDocumentList &batch_result = context->result(q);
    ASSERT_EQ(topk, batch_result.size());

    auto single_ctx = searcher.create_context();
    ASSERT_TRUE(!!single_ctx);
    single_ctx->set_topk(topk);
    ASSERT_EQ(0, searcher.search_impl(&query_data_[q * dimension_], qmeta,
                                      single_ctx));
    const IndexDocumentList &single_result = single_ctx->result(0);
    ASSERT_EQ(topk, single_result.size());
    for (size_t i = 0; i < topk; ++i) {
      EXPECT_EQ(single_result[i].key(), batch_result[i].key());
      EXPECT_NEAR(single_result[i].score(), batch_result[i].score(), 1e-5f);
    }
  }
}

TEST_F(IVFTurboQuantizerTest, TestLegacyIndexNotAffected) {
  //! Plain IVF without turbo params must still build/load/search.
  Params params;
  params.set(PARAM_IVF_BUILDER_CENTROID_COUNT, "16");
  params.set(PARAM_IVF_BUILDER_CLUSTER_CLASS, "KmeansCluster");

  IVFSearcher searcher;
  IndexStorage::Pointer container;
  build_dump_load(params, &searcher, &container);

  auto context = searcher.create_context();
  ASSERT_TRUE(!!context);
  context->set_topk(1);
  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dimension_);
  ASSERT_EQ(0, searcher.search_impl(query_data_.data(), qmeta, context));
  const IndexDocumentList &result = context->result(0);
  ASSERT_EQ((size_t)1, result.size());

  //! Exact fp32 distances: top1 must match brute force.
  auto gt = brute_force_topk(query_data_.data(), 1, false);
  EXPECT_EQ(gt[0], result[0].key());
}

