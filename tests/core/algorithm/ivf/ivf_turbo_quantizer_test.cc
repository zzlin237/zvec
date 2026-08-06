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

  //! Load the previously dumped index into another searcher, with extra
  //! searcher params (e.g. the precompute switch).
  void load_dumped(IVFSearcher *searcher, const Params &extra_params,
                   IndexStorage::Pointer *container);

  //! Generate well-separated clusters so residual quantization can show
  //! its advantage over raw-vector quantization.
  void prepare_clustered_data();

  //! Brute-force ground truth ids for the stored base vectors.
  vector<uint64_t> brute_force_topk(const float *query, size_t topk,
                                    bool cosine) const;

  void recall_at(const Params &builder_params, size_t topk, bool cosine,
                 float *recall);

  //! Recall against the stored base vectors using an already loaded (or
  //! in-memory) searcher.
  float recall_with_searcher(IVFSearcher *searcher, size_t topk, bool cosine);

  //! Search all queries and collect (key, score) pairs per query.
  void search_scored(IVFSearcher *searcher, size_t topk,
                     vector<vector<pair<uint64_t, float>>> *results);

  uint32_t dimension_{32};
  uint32_t base_num_{1000};
  uint32_t query_num_{32};
  vector<float> base_data_{};
  vector<float> query_data_{};
  string index_path_{};
  IndexHolder::Pointer holder_{};
  std::string build_metric_{};  //! metric override for build_dump_load
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
  auto holder =
      std::make_shared<MultiPassIndexHolder<IndexMeta::DataType::DT_FP32>>(
          dimension_);
  for (uint32_t i = 0; i < num; ++i) {
    NumericalVector<float> vec(dimension_);
    for (uint32_t j = 0; j < dimension_; ++j) {
      vec[j] = base_data_[i * dimension_ + j];
    }
    holder->emplace(i, vec);
  }
  holder_ = holder;
}

void IVFTurboQuantizerTest::prepare_clustered_data() {
  //! 16 well-separated Gaussian clusters: residuals are tiny compared to
  //! raw vectors, so residual PQ should clearly beat raw PQ.
  std::mt19937 gen(54321);
  std::uniform_real_distribution<float> center_dist(-8.0f, 8.0f);
  std::normal_distribution<float> noise(0.0f, 0.2f);
  const uint32_t cluster_num = 16;
  vector<float> centers(cluster_num * dimension_);
  for (auto &v : centers) {
    v = center_dist(gen);
  }
  base_data_.resize(base_num_ * dimension_);
  for (uint32_t i = 0; i < base_num_; ++i) {
    const float *center = &centers[(i % cluster_num) * dimension_];
    for (uint32_t j = 0; j < dimension_; ++j) {
      base_data_[i * dimension_ + j] = center[j] + noise(gen);
    }
  }
}

void IVFTurboQuantizerTest::build_dump_load(const Params &builder_params,
                                            IVFSearcher *searcher,
                                            IndexStorage::Pointer *container) {
  IndexMeta meta;
  meta.set_meta(IndexMeta::DataType::DT_FP32, dimension_);
  std::string metric = build_metric_;
  if (metric.empty()) {
    metric = builder_params.has(PARAM_IVF_BUILDER_CONVERTER_CLASS)
                 ? "Cosine"
                 : "SquaredEuclidean";
  }
  meta.set_metric(metric, 0, Params());

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

void IVFTurboQuantizerTest::load_dumped(IVFSearcher *searcher,
                                        const Params &extra_params,
                                        IndexStorage::Pointer *container) {
  Params searcher_params;
  searcher_params.set(PARAM_IVF_SEARCHER_SCAN_RATIO, 1.0);
  searcher_params.set(PARAM_IVF_SEARCHER_BRUTE_FORCE_THRESHOLD, 0);
  searcher_params.merge(extra_params);
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

void IVFTurboQuantizerTest::recall_at(const Params &builder_params, size_t topk,
                                      bool cosine, float *recall) {
  IVFSearcher searcher;
  IndexStorage::Pointer container;
  build_dump_load(builder_params, &searcher, &container);
  *recall = this->recall_with_searcher(&searcher, topk, cosine);
}

float IVFTurboQuantizerTest::recall_with_searcher(IVFSearcher *searcher,
                                                  size_t topk, bool cosine) {
  auto context = searcher->create_context();
  EXPECT_TRUE(!!context);
  context->set_topk(topk);

  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dimension_);
  size_t hit = 0;
  for (uint32_t q = 0; q < query_num_; ++q) {
    const float *query = &query_data_[q * dimension_];
    EXPECT_EQ(0, searcher->search_impl(query, qmeta, context));
    const IndexDocumentList &result = context->result(0);
    EXPECT_EQ(topk, result.size());
    auto gt = brute_force_topk(query, topk, cosine);
    std::set<uint64_t> gt_set(gt.begin(), gt.end());
    for (size_t i = 0; i < result.size(); ++i) {
      if (gt_set.count(result[i].key())) {
        ++hit;
      }
    }
  }
  return static_cast<float>(hit) / (query_num_ * topk);
}

void IVFTurboQuantizerTest::search_scored(
    IVFSearcher *searcher, size_t topk,
    vector<vector<pair<uint64_t, float>>> *results) {
  auto context = searcher->create_context();
  EXPECT_TRUE(!!context);
  context->set_topk(topk);

  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dimension_);
  results->assign(query_num_, {});
  for (uint32_t q = 0; q < query_num_; ++q) {
    const float *query = &query_data_[q * dimension_];
    EXPECT_EQ(0, searcher->search_impl(query, qmeta, context));
    const IndexDocumentList &result = context->result(0);
    EXPECT_EQ(topk, result.size());
    for (size_t i = 0; i < result.size(); ++i) {
      (*results)[q].emplace_back(result[i].key(), result[i].score());
    }
  }
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
  ASSERT_EQ(
      0, searcher.search_impl(query_data_.data(), qmeta, query_num_, context));
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

namespace {
Params make_turbo_params(bool use_residual) {
  Params params;
  params.set(PARAM_IVF_BUILDER_CENTROID_COUNT, "16");
  params.set(PARAM_IVF_BUILDER_CLUSTER_CLASS, "KmeansCluster");
  params.set(PARAM_IVF_BUILDER_TURBO_QUANTIZER_CLASS, "PqInt8Quantizer");
  params.set("num_chunk", 8);
  if (use_residual) {
    params.set(PARAM_IVF_BUILDER_USE_RESIDUAL, true);
  }
  return params;
}
}  // namespace

TEST_F(IVFTurboQuantizerTest, TestL2PqInt8Residual) {
  //! On well-separated clusters the residual vectors are much smaller than
  //! the raw vectors, so residual PQ must not lose recall against raw PQ.
  //! Fewer vectors per cluster widen the top-k distance gaps relative to
  //! the PQ distortion of the (tiny) residuals.
  base_num_ = 320;
  prepare_clustered_data();

  float recall_plain = 0.0f;
  recall_at(make_turbo_params(false), 10, false, &recall_plain);

  float recall_residual = 0.0f;
  recall_at(make_turbo_params(true), 10, false, &recall_residual);

  //! IVF contract: with the same quantizer and the same data, feeding
  //! residuals instead of raw vectors must not lose recall. The absolute
  //! floor only guards the driver quantizer's baseline quality.
  EXPECT_GE(recall_residual, recall_plain);
  EXPECT_GT(recall_residual, 0.85f);
}

TEST_F(IVFTurboQuantizerTest, TestCosinePqInt8Residual) {
  //! Cosine residual mode: IVF normalizes internally (no converter), the
  //! quantizer works in the residual space with an L2 metric.
  base_num_ = 320;
  prepare_clustered_data();
  build_metric_ = "Cosine";

  float recall_plain = 0.0f;
  recall_at(make_turbo_params(false), 10, true, &recall_plain);

  float recall_residual = 0.0f;
  recall_at(make_turbo_params(true), 10, true, &recall_residual);

  //! On normalized data raw PQ already works well; residual must stay on
  //! par (the gap is within the kmeans/PQ run-to-run noise).
  EXPECT_GE(recall_residual, recall_plain - 0.05f);
  EXPECT_GT(recall_residual, 0.85f);

  //! Score semantics: residual scores must be cosine distances (1 - cos).
  IVFSearcher searcher;
  IndexStorage::Pointer container;
  build_dump_load(make_turbo_params(true), &searcher, &container);
  vector<vector<pair<uint64_t, float>>> results;
  search_scored(&searcher, 10, &results);
  for (uint32_t q = 0; q < query_num_; ++q) {
    const float *query = &query_data_[q * dimension_];
    double qnorm = 0.0;
    for (uint32_t j = 0; j < dimension_; ++j) {
      qnorm += query[j] * query[j];
    }
    for (const auto &it : results[q]) {
      const float *vec = &base_data_[it.first * dimension_];
      double dot = 0.0, vnorm = 0.0;
      for (uint32_t j = 0; j < dimension_; ++j) {
        dot += query[j] * vec[j];
        vnorm += vec[j] * vec[j];
      }
      const double denom = std::sqrt(qnorm * vnorm);
      const float cos_dist =
          static_cast<float>(denom == 0.0 ? 1.0 : 1.0 - dot / denom);
      //! Loose tolerance: scores are PQ-approximated cosine distances.
      EXPECT_NEAR(it.second, cos_dist, 0.25f);
    }
  }
}

TEST_F(IVFTurboQuantizerTest, TestResidualRestore) {
  //! A dumped residual index must restore the use_residual switch and the
  //! quantizer metric conversion (L2) through the meta: reloading the same
  //! file must reproduce the exact same results with high recall.
  base_num_ = 320;
  prepare_clustered_data();
  const size_t topk = 10;

  for (int cosine = 0; cosine <= 1; ++cosine) {
    build_metric_ = cosine ? "Cosine" : "SquaredEuclidean";
    Params params = make_turbo_params(true);

    IVFSearcher searcher;
    IndexStorage::Pointer container;
    build_dump_load(params, &searcher, &container);
    const float recall = recall_with_searcher(&searcher, topk, cosine != 0);
    EXPECT_GT(recall, 0.85f) << "metric=" << build_metric_;

    vector<vector<pair<uint64_t, float>>> first;
    search_scored(&searcher, topk, &first);
    searcher.cleanup();
    container.reset();

    //! Reload the dumped file from scratch.
    IVFSearcher reloaded;
    Params searcher_params;
    searcher_params.set(PARAM_IVF_SEARCHER_SCAN_RATIO, 1.0);
    searcher_params.set(PARAM_IVF_SEARCHER_BRUTE_FORCE_THRESHOLD, 0);
    ASSERT_EQ(0, reloaded.init(searcher_params));
    IndexStorage::Pointer container2 =
        IndexFactory::CreateStorage("MMapFileReadStorage");
    ASSERT_TRUE(!!container2);
    Params container_params;
    container_params.set("proxima.mmap_file.container.memory_warmup", true);
    container2->init(container_params);
    ASSERT_EQ(0, container2->open(index_path_, false));
    ASSERT_EQ(0, reloaded.load(container2, IndexMetric::Pointer()));

    vector<vector<pair<uint64_t, float>>> second;
    search_scored(&reloaded, topk, &second);
    ASSERT_EQ(first.size(), second.size());
    for (size_t q = 0; q < first.size(); ++q) {
      ASSERT_EQ(first[q].size(), second[q].size());
      for (size_t i = 0; i < first[q].size(); ++i) {
        EXPECT_EQ(first[q][i].first, second[q][i].first);
        EXPECT_NEAR(first[q][i].second, second[q][i].second, 1e-5f);
      }
    }
  }
}

TEST_F(IVFTurboQuantizerTest, TestPrecomputeEquivalence) {
  //! The precomputed residual table (term2/term3 decomposition) must not
  //! change results: searchers with the table on/off over the same dump
  //! return identical keys and nearly equal scores (the merge only
  //! reshuffles float rounding against the per-list LUT path).
  base_num_ = 320;
  prepare_clustered_data();
  const size_t topk = 10;

  for (int cosine = 0; cosine <= 1; ++cosine) {
    build_metric_ = cosine ? "Cosine" : "SquaredEuclidean";

    IVFSearcher on_searcher;
    IndexStorage::Pointer on_container;
    build_dump_load(make_turbo_params(true), &on_searcher, &on_container);

    Params off_params;
    off_params.set(PARAM_IVF_SEARCHER_USE_PRECOMPUTE_TABLE, false);
    IVFSearcher off_searcher;
    IndexStorage::Pointer off_container;
    load_dumped(&off_searcher, off_params, &off_container);

    vector<vector<pair<uint64_t, float>>> on_results, off_results;
    search_scored(&on_searcher, topk, &on_results);
    search_scored(&off_searcher, topk, &off_results);

    ASSERT_EQ(on_results.size(), off_results.size());
    for (size_t q = 0; q < on_results.size(); ++q) {
      ASSERT_EQ(on_results[q].size(), off_results[q].size());
      for (size_t i = 0; i < topk; ++i) {
        EXPECT_EQ(on_results[q][i].first, off_results[q][i].first)
            << "metric=" << build_metric_ << " q=" << q << " i=" << i;
        EXPECT_NEAR(on_results[q][i].second, off_results[q][i].second, 1e-3f);
      }
    }

    //! The accelerated path keeps the residual recall level.
    const float recall = recall_with_searcher(&on_searcher, topk, cosine != 0);
    EXPECT_GT(recall, 0.9f) << "metric=" << build_metric_;

    on_searcher.cleanup();
    off_searcher.cleanup();
    on_container.reset();
    off_container.reset();
  }
}

TEST_F(IVFTurboQuantizerTest, TestPrecomputeRestore) {
  //! The table is not persisted: after dump/load it must be rebuilt from
  //! the residual centroid segment automatically, reproducing results.
  base_num_ = 320;
  prepare_clustered_data();
  const size_t topk = 10;

  for (int cosine = 0; cosine <= 1; ++cosine) {
    build_metric_ = cosine ? "Cosine" : "SquaredEuclidean";

    IVFSearcher searcher;
    IndexStorage::Pointer container;
    build_dump_load(make_turbo_params(true), &searcher, &container);
    vector<vector<pair<uint64_t, float>>> first;
    search_scored(&searcher, topk, &first);
    EXPECT_GT(recall_with_searcher(&searcher, topk, cosine != 0), 0.9f)
        << "metric=" << build_metric_;
    searcher.cleanup();
    container.reset();

    //! Reload the dumped file; the table is rebuilt on load.
    IVFSearcher reloaded;
    IndexStorage::Pointer container2;
    load_dumped(&reloaded, Params(), &container2);
    vector<vector<pair<uint64_t, float>>> second;
    search_scored(&reloaded, topk, &second);

    ASSERT_EQ(first.size(), second.size());
    for (size_t q = 0; q < first.size(); ++q) {
      ASSERT_EQ(first[q].size(), second[q].size());
      for (size_t i = 0; i < first[q].size(); ++i) {
        EXPECT_EQ(first[q][i].first, second[q][i].first);
        EXPECT_NEAR(first[q][i].second, second[q][i].second, 1e-5f);
      }
    }
    EXPECT_GT(recall_with_searcher(&reloaded, topk, cosine != 0), 0.9f)
        << "metric=" << build_metric_;

    reloaded.cleanup();
    container2.reset();
  }
}

TEST_F(IVFTurboQuantizerTest, TestPrecomputeFallback) {
  //! PqInt4Quantizer also implements the precomputed table protocol: the
  //! on/off searchers over the same dump must return identical keys with
  //! nearly equal scores, mirroring TestPrecomputeEquivalence for int8.
  base_num_ = 320;
  prepare_clustered_data();
  const size_t topk = 10;

  Params params = make_turbo_params(true);
  params.set(PARAM_IVF_BUILDER_TURBO_QUANTIZER_CLASS, "PqInt4Quantizer");

  IVFSearcher searcher;
  IndexStorage::Pointer container;
  build_dump_load(params, &searcher, &container);

  Params off_params;
  off_params.set(PARAM_IVF_SEARCHER_USE_PRECOMPUTE_TABLE, false);
  IVFSearcher off_searcher;
  IndexStorage::Pointer off_container;
  load_dumped(&off_searcher, off_params, &off_container);

  vector<vector<pair<uint64_t, float>>> on_results, off_results;
  search_scored(&searcher, topk, &on_results);
  search_scored(&off_searcher, topk, &off_results);

  ASSERT_EQ(on_results.size(), off_results.size());
  for (size_t q = 0; q < on_results.size(); ++q) {
    ASSERT_EQ(on_results[q].size(), off_results[q].size());
    for (size_t i = 0; i < topk; ++i) {
      //! Precomputed merge only reshuffles float rounding against the
      //! per-list LUT path: keys identical, scores nearly equal.
      EXPECT_EQ(on_results[q][i].first, off_results[q][i].first);
      EXPECT_NEAR(on_results[q][i].second, off_results[q][i].second, 1e-3f);
    }
  }

  //! Loose floor: int4 residuals are much coarser than int8; the point of
  //! this test is the on/off equivalence above, this only guards a breakdown.
  EXPECT_GT(recall_with_searcher(&searcher, topk, false), 0.35f);
}

TEST_F(IVFTurboQuantizerTest, TestResidualRejectIp) {
  //! Residuals break inner-product ordering, IP must be rejected.
  IndexMeta ip_meta;
  ip_meta.set_meta(IndexMeta::DataType::DT_FP32, dimension_);
  ip_meta.set_metric("InnerProduct", 0, Params());

  IVFBuilder builder;
  EXPECT_NE(0, builder.init(ip_meta, make_turbo_params(true)));

  //! The converter chain is orthogonal to residual and must be rejected.
  IndexMeta l2_meta;
  l2_meta.set_meta(IndexMeta::DataType::DT_FP32, dimension_);
  l2_meta.set_metric("SquaredEuclidean", 0, Params());
  Params conv_params = make_turbo_params(true);
  conv_params.set(PARAM_IVF_BUILDER_CONVERTER_CLASS,
                  "CosineNormalizeConverter");
  IVFBuilder builder2;
  EXPECT_NE(0, builder2.init(l2_meta, conv_params));
}

namespace {
Params make_pq_fast_params(bool use_residual) {
  Params params;
  params.set(PARAM_IVF_BUILDER_CENTROID_COUNT, "16");
  params.set(PARAM_IVF_BUILDER_CLUSTER_CLASS, "KmeansCluster");
  params.set(PARAM_IVF_BUILDER_TURBO_QUANTIZER_CLASS, "PqFastQuantizer");
  params.set("num_chunk", 8);
  if (use_residual) {
    params.set(PARAM_IVF_BUILDER_USE_RESIDUAL, true);
  }
  return params;
}
}  // namespace

TEST_F(IVFTurboQuantizerTest, TestL2PqFast) {
  //! FastScan stores codes in packed 32-vector blocks; plain L2 recall
  //! must stay on par with the gather-style 4-bit PQ (the extra u8 LUT
  //! affine quantization only costs a little).
  Params int4_params;
  int4_params.set(PARAM_IVF_BUILDER_CENTROID_COUNT, "16");
  int4_params.set(PARAM_IVF_BUILDER_CLUSTER_CLASS, "KmeansCluster");
  int4_params.set(PARAM_IVF_BUILDER_TURBO_QUANTIZER_CLASS, "PqInt4Quantizer");
  int4_params.set("num_chunk", 8);
  float recall_int4 = 0.0f;
  recall_at(int4_params, 10, false, &recall_int4);

  float recall_fast = 0.0f;
  recall_at(make_pq_fast_params(false), 10, false, &recall_fast);

  EXPECT_GT(recall_fast, recall_int4 - 0.1f);
  EXPECT_GT(recall_fast, 0.3f);
}

TEST_F(IVFTurboQuantizerTest, TestL2PqFastResidualPrecompute) {
  //! Residual + precomputed table on/off equivalence for the packed-u8
  //! FastScan path.  Both paths affine-quantize the LUT to u8, so scores
  //! only agree within the combined rounding tolerance (looser than the
  //! float-table quantizers).
  base_num_ = 320;
  prepare_clustered_data();
  const size_t topk = 10;

  IVFSearcher on_searcher;
  IndexStorage::Pointer on_container;
  build_dump_load(make_pq_fast_params(true), &on_searcher, &on_container);

  Params off_params;
  off_params.set(PARAM_IVF_SEARCHER_USE_PRECOMPUTE_TABLE, false);
  IVFSearcher off_searcher;
  IndexStorage::Pointer off_container;
  load_dumped(&off_searcher, off_params, &off_container);

  vector<vector<pair<uint64_t, float>>> on_results, off_results;
  search_scored(&on_searcher, topk, &on_results);
  search_scored(&off_searcher, topk, &off_results);

  //! Both paths affine-quantize the LUT to u8 with different delta/bias,
  //! so near-tie keys may swap; the sorted score lists must still agree
  //! pointwise within the combined rounding bound (relative to the score
  //! magnitude), and key overlap must stay high.
  ASSERT_EQ(on_results.size(), off_results.size());
  for (size_t q = 0; q < on_results.size(); ++q) {
    ASSERT_EQ(on_results[q].size(), off_results[q].size());
    for (size_t i = 0; i < topk; ++i) {
      const float tol = 0.005f * std::abs(off_results[q][i].second) + 0.5f;
      EXPECT_NEAR(on_results[q][i].second, off_results[q][i].second, tol)
          << "q=" << q << " i=" << i;
    }
  }
  size_t set_overlap = 0;
  for (size_t q = 0; q < on_results.size(); ++q) {
    std::set<uint64_t> off_keys;
    for (const auto &it : off_results[q]) {
      off_keys.insert(it.first);
    }
    for (size_t i = 0; i < topk; ++i) {
      if (off_keys.count(on_results[q][i].first)) {
        ++set_overlap;
      }
    }
  }
  //! Measured ~0.93; the floor only guards a breakdown.
  EXPECT_GT(static_cast<float>(set_overlap), 0.7f * on_results.size() * topk);

  //! Loose floor: the point of this test is the on/off equivalence above;
  //! this only guards a breakdown (same level as the int4 residual test).
  EXPECT_GT(recall_with_searcher(&on_searcher, topk, false), 0.35f);

  on_searcher.cleanup();
  off_searcher.cleanup();
}

TEST_F(IVFTurboQuantizerTest, TestPqFastRejectsSmallBlock) {
  //! Packed FastScan blocks interleave exactly 32 codes: any other
  //! block_vector_count must be rejected at dump time.
  prepare_holder(base_num_);

  Params params = make_pq_fast_params(false);
  params.set(PARAM_IVF_BUILDER_BLOCK_VECTOR_COUNT, 16);

  IndexMeta meta;
  meta.set_meta(IndexMeta::DataType::DT_FP32, dimension_);
  meta.set_metric("SquaredEuclidean", 0, Params());

  IVFBuilder builder;
  ASSERT_EQ(0, builder.init(meta, params));
  ASSERT_EQ(0, builder.train(nullptr, holder_));
  ASSERT_EQ(0, builder.build(nullptr, holder_));

  IndexDumper::Pointer dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_TRUE(!!dumper);
  ASSERT_EQ(0, dumper->create(index_path_));
  EXPECT_NE(0, builder.dump(dumper));
  dumper->close();
}
