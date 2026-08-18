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

// Tests for the quantized graph (QG) region of HnswStreamer: each node record
// also stores the packed codes of its level 0 neighbors, materialized on
// flush/close, after which a graph hop scans one block instead of gathering a
// vector per candidate.
#include "hnsw_qg.h"
#include <cstring>
#include <random>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <turbo/quantizer/quantizer.h>
#include <zvec/ailego/container/params.h>
#include <zvec/ailego/container/vector.h>
#include <zvec/core/framework/index_factory.h>
#include "tests/test_util.h"
#include "hnsw_params.h"
#include "hnsw_streamer.h"

using namespace std;
using namespace testing;
using namespace zvec::ailego;

namespace zvec {
namespace core {

constexpr size_t kDim = 32;
constexpr size_t kNumChunk = 8;
constexpr size_t kTrainCount = 4000;
constexpr size_t kDocCount = 2000;
constexpr size_t kQueryCount = 30;
constexpr uint32_t kTopk = 10;

class HnswQgTest : public testing::Test {
 protected:
  void SetUp(void) override {
    zvec::test_util::RemoveTestPath(dir_);
  }

  void TearDown(void) override {
    zvec::test_util::RemoveTestPath(dir_);
  }

  static std::string dir_;
};

std::string HnswQgTest::dir_("hnsw_qg_test_dir/");

static turbo::Quantizer::Pointer make_fast_quantizer() {
  auto quantizer = IndexFactory::CreateQuantizer("PqFastQuantizer");
  EXPECT_TRUE(quantizer != nullptr);
  if (!quantizer) {
    return nullptr;
  }
  IndexMeta meta;
  meta.set_meta(IndexMeta::DataType::DT_FP32, kDim);
  meta.set_metric("SquaredEuclidean", 0, ailego::Params());
  ailego::Params qparams;
  qparams.set("num_chunk", static_cast<uint32_t>(kNumChunk));
  EXPECT_EQ(0, quantizer->init(meta, qparams));
  return quantizer;
}

static std::shared_ptr<MultiPassIndexHolder<IndexMeta::DataType::DT_FP32>>
make_train_holder(uint32_t seed) {
  auto holder =
      std::make_shared<MultiPassIndexHolder<IndexMeta::DataType::DT_FP32>>(
          kDim);
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (size_t i = 0; i < kTrainCount; ++i) {
    NumericalVector<float> vec(kDim);
    for (size_t j = 0; j < kDim; ++j) {
      vec[j] = dist(gen);
    }
    holder->emplace(i + 1, vec);
  }
  return holder;
}

static void set_hnsw_params(ailego::Params *params, bool qg_enable) {
  params->set(PARAM_HNSW_STREAMER_MAX_NEIGHBOR_COUNT, 16U);
  params->set(PARAM_HNSW_STREAMER_SCALING_FACTOR, 5U);
  params->set(PARAM_HNSW_STREAMER_EFCONSTRUCTION, 100U);
  params->set(PARAM_HNSW_STREAMER_EF, 200U);
  params->set(PARAM_HNSW_STREAMER_GET_VECTOR_ENABLE, true);
  params->set(PARAM_HNSW_STREAMER_QG_ENABLE, qg_enable);
  //! Construction params of the quantizer, as the build configs pass them:
  //! a reopen recreates the quantizer from these before loading its codebook.
  params->set("num_chunk", static_cast<uint32_t>(kNumChunk));
}

static vector<pair<uint64_t, float>> collect_scored(
    IndexStreamer::Context::Pointer &ctx) {
  vector<pair<uint64_t, float>> scored;
  auto &result = ctx->result();
  scored.reserve(result.size());
  for (size_t i = 0; i < result.size(); ++i) {
    scored.emplace_back(result[i].key(), result[i].score());
  }
  return scored;
}

static vector<uint64_t> keys_of(const vector<pair<uint64_t, float>> &scored) {
  vector<uint64_t> keys;
  keys.reserve(scored.size());
  for (const auto &s : scored) {
    keys.push_back(s.first);
  }
  return keys;
}

static float calc_recall(const vector<uint64_t> &keys,
                         const vector<uint64_t> &bf_keys) {
  std::set<uint64_t> bf_set(bf_keys.begin(), bf_keys.end());
  size_t hit = 0;
  for (auto key : keys) {
    if (bf_set.count(key)) {
      ++hit;
    }
  }
  return static_cast<float>(hit) /
         static_cast<float>(std::max<size_t>(1UL, bf_keys.size()));
}

//! The block scan and the per-candidate scan share codes and query LUT, so a
//! key returned by both must carry the same distance.  Returns the number of
//! keys compared, so the caller can require that comparison actually happened
//! across the query set (guarding a wrongly packed or addressed region, which
//! a recall check alone would miss).
static size_t count_score_matches(const vector<pair<uint64_t, float>> &graph,
                                  const vector<pair<uint64_t, float>> &bf) {
  size_t compared = 0;
  for (const auto &g : graph) {
    for (const auto &b : bf) {
      if (g.first == b.first) {
        EXPECT_NEAR(b.second, g.second, 1e-3f) << "key=" << g.first;
        ++compared;
        break;
      }
    }
  }
  return compared;
}

//! Run the query set through both the graph (block scan) and brute force
//! paths, returning the average recall and accumulating the number of
//! score comparisons across all queries.
static float run_queries(IndexStreamer::Pointer &streamer,
                         const vector<NumericalVector<float>> &queries,
                         const IndexQueryMeta &qmeta, size_t *total_compared) {
  auto ctx = streamer->create_context();
  EXPECT_TRUE(!!ctx);
  float recall_sum = 0.0f;
  for (size_t q = 0; q < queries.size(); ++q) {
    ctx->set_topk(kTopk);
    EXPECT_EQ(0, streamer->search_impl(queries[q].data(), qmeta, ctx));
    vector<pair<uint64_t, float>> graph_scored = collect_scored(ctx);
    EXPECT_EQ(kTopk, graph_scored.size());

    EXPECT_EQ(0, streamer->search_bf_impl(queries[q].data(), qmeta, ctx));
    vector<pair<uint64_t, float>> bf_scored = collect_scored(ctx);
    EXPECT_EQ(kTopk, bf_scored.size());

    *total_compared += count_score_matches(graph_scored, bf_scored);
    recall_sum += calc_recall(keys_of(graph_scored), keys_of(bf_scored));
  }
  return recall_sum / static_cast<float>(queries.size());
}

//! Geometry of the region: placement, alignment and block count.
TEST_F(HnswQgTest, TestLayoutGeometry) {
  QgLayout layout;
  //! Disabled: no region, plain 32-byte record alignment.
  EXPECT_FALSE(layout.enabled());
  EXPECT_EQ(0UL, layout.blocks_per_node(64));
  EXPECT_EQ(0UL, layout.region_bytes(64));
  EXPECT_EQ(64UL, layout.configure(33UL, 64UL));
  EXPECT_FALSE(layout.enabled());
  EXPECT_EQ(0U, layout.offset);

  //! Enabled: 32 vectors per 128-byte block.
  layout.block_vectors = 32U;
  layout.block_bytes = 128U;
  EXPECT_EQ(1UL, layout.blocks_per_node(1));
  EXPECT_EQ(1UL, layout.blocks_per_node(32));
  EXPECT_EQ(2UL, layout.blocks_per_node(33));
  EXPECT_EQ(2UL, layout.blocks_per_node(64));
  EXPECT_EQ(256UL, layout.region_bytes(64));

  //! The region starts cache-line aligned after the base record, and the node
  //! size stays cache-line aligned as well.
  const size_t node_size = layout.configure(100UL, 64UL);
  EXPECT_TRUE(layout.enabled());
  EXPECT_EQ(128U, layout.offset);
  EXPECT_EQ(0U, layout.offset % 64U);
  EXPECT_EQ(256U, layout.region_size);
  EXPECT_EQ(128UL + 256UL, node_size);
  EXPECT_EQ(0UL, node_size % 64UL);
}

//! End to end: build with a provider of original vectors (FastScan has no
//! code-vs-code distance, so the graph is built on the originals), materialize
//! on flush, then search through the block scan and check recall against the
//! per-candidate brute force on the same codes.  Reopening restores the region
//! geometry and the quantizer from the persisted meta, so the scan keeps
//! working with no quantizer injected.
TEST_F(HnswQgTest, TestQgSearchRecall) {
  auto quantizer = make_fast_quantizer();
  ASSERT_TRUE(quantizer != nullptr);
  ASSERT_EQ(0, quantizer->train(make_train_holder(42)));

  std::mt19937 gen(7);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  //! FastScan now provides SDC, so the graph is built on the codes
  //! (code-vs-code), exactly like PqInt4 -- no provider of original vectors.
  vector<NumericalVector<float>> raw_vecs(kDocCount,
                                          NumericalVector<float>(kDim));
  for (size_t i = 0; i < kDocCount; ++i) {
    for (size_t j = 0; j < kDim; ++j) {
      raw_vecs[i][j] = dist(gen);
    }
  }

  IndexMeta meta;
  meta.set_meta(IndexMeta::DataType::DT_FP32, kDim);
  meta.set_metric("SquaredEuclidean", 0, ailego::Params());
  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32,
                       static_cast<uint32_t>(kDim));

  const string index_path = dir_ + "qg.index";
  vector<NumericalVector<float>> queries(kQueryCount,
                                         NumericalVector<float>(kDim));
  for (size_t q = 0; q < kQueryCount; ++q) {
    for (size_t j = 0; j < kDim; ++j) {
      queries[q][j] = dist(gen);
    }
  }

  size_t total_compared = 0;
  {
    auto streamer = IndexFactory::CreateStreamer("HnswStreamer");
    ASSERT_TRUE(streamer != nullptr);
    auto hnsw = std::dynamic_pointer_cast<HnswStreamer>(streamer);
    ASSERT_TRUE(hnsw != nullptr);

    ailego::Params params;
    set_hnsw_params(&params, /*qg_enable=*/true);
    ASSERT_EQ(0, streamer->init(meta, params));
    ASSERT_EQ(0, streamer->init_quantizer(quantizer));

    auto storage = IndexFactory::CreateStorage("MMapFileStorage");
    ASSERT_TRUE(storage != nullptr);
    ailego::Params stg_params;
    ASSERT_EQ(0, storage->init(stg_params));
    ASSERT_EQ(0, storage->open(index_path, true));
    ASSERT_EQ(0, streamer->open(storage));

    auto add_ctx = streamer->create_context();
    ASSERT_TRUE(!!add_ctx);
    for (size_t i = 0; i < kDocCount; ++i) {
      ASSERT_EQ(0, streamer->add_impl(i, raw_vecs[i].data(), qmeta, add_ctx));
    }

    //! Materializes the region, after which the index is read-only.
    ASSERT_EQ(0, streamer->flush(0UL));
    ASSERT_TRUE(hnsw->qg_ready());
    ASSERT_EQ(
        IndexError_Unsupported,
        streamer->add_impl(kDocCount, raw_vecs[0].data(), qmeta, add_ctx));

    float avg_recall = run_queries(streamer, queries, qmeta, &total_compared);
    EXPECT_GE(avg_recall, 0.8f) << "avg_recall=" << avg_recall;

    ASSERT_EQ(0, streamer->close());
  }
  //! The block scan and per-candidate scan agreed on distances for shared
  //! keys across the query set (not merely produced similar recall).
  EXPECT_GT(total_compared, kQueryCount);

  //! Reopen without injecting a quantizer: the region geometry and the
  //! quantizer both come from the persisted meta.
  {
    auto streamer = IndexFactory::CreateStreamer("HnswStreamer");
    ASSERT_TRUE(streamer != nullptr);
    auto hnsw = std::dynamic_pointer_cast<HnswStreamer>(streamer);
    ASSERT_TRUE(hnsw != nullptr);

    ailego::Params params;
    set_hnsw_params(&params, /*qg_enable=*/true);
    ASSERT_EQ(0, streamer->init(meta, params));

    auto storage = IndexFactory::CreateStorage("MMapFileStorage");
    ASSERT_TRUE(storage != nullptr);
    ailego::Params stg_params;
    ASSERT_EQ(0, storage->init(stg_params));
    ASSERT_EQ(0, storage->open(index_path, false));
    ASSERT_EQ(0, streamer->open(storage));

    //! Without either the region or the quantizer the block scan would
    //! silently not run.
    ASSERT_TRUE(hnsw->qg_ready());
    ASSERT_TRUE(hnsw->quantizer() != nullptr);

    size_t reopened_compared = 0;
    float avg_recall =
        run_queries(streamer, queries, qmeta, &reopened_compared);
    EXPECT_GE(avg_recall, 0.8f) << "reopened avg_recall=" << avg_recall;
    EXPECT_GT(reopened_compared, kQueryCount);
  }
}

//! FastScan now supports SDC (code-vs-code), so it builds the graph on the
//! codes like PqInt4, with no provider of original vectors: distinct codes
//! give a non-negative distance and a code against itself gives ~0.
TEST_F(HnswQgTest, TestSupportsSdc) {
  auto quantizer = make_fast_quantizer();
  ASSERT_TRUE(quantizer != nullptr);
  ASSERT_EQ(0, quantizer->train(make_train_holder(42)));
  ASSERT_TRUE(quantizer->supports_sdc());

  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32,
                       static_cast<uint32_t>(kDim));
  std::mt19937 gen(11);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  const size_t code_len = quantizer->quantized_datapoint_vector_length();
  vector<uint8_t> code_a(code_len), code_b(code_len);
  NumericalVector<float> va(kDim), vb(kDim);
  for (size_t j = 0; j < kDim; ++j) {
    va[j] = dist(gen);
    vb[j] = dist(gen);
  }
  quantizer->quantize_data(va.data(), code_a.data());
  quantizer->quantize_data(vb.data(), code_b.data());

  //! Code against itself is (near) zero; two distinct codes are non-negative.
  EXPECT_NEAR(0.0f,
              quantizer->calc_distance_dp_dp(code_a.data(), code_a.data()),
              1e-3f);
  EXPECT_GE(quantizer->calc_distance_dp_dp(code_a.data(), code_b.data()), 0.0f);
}

}  // namespace core
}  // namespace zvec
