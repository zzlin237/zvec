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

// Tests for HnswStreamer with a turbo::Quantizer attached:
// - PqInt8Quantizer: datapoints stored as PQ codes, search with an
//   ADC LUT query, graph built with SDC distances.
// - Fp32Quantizer: pass-through quantizer exercising the default
//   quantizer path with fp32 storage.
#include "hnsw_streamer.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <set>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <zvec/ailego/container/params.h>
#include <zvec/ailego/container/vector.h>
#include <zvec/core/framework/index_factory.h>
#include <turbo/quantizer/quantizer.h>
#include "metric/metric_params.h"
#include "tests/test_util.h"

using namespace std;
using namespace testing;
using namespace zvec::ailego;

namespace zvec {
namespace core {

constexpr size_t kDim = 16;
constexpr size_t kNumChunk = 4;
constexpr size_t kTrainCount = 4000;
constexpr size_t kDocCount = 2000;
constexpr size_t kQueryCount = 50;
constexpr uint32_t kTopk = 10;

class HnswStreamerTurboPqTest : public testing::Test {
 protected:
  void SetUp(void) override {
    zvec::test_util::RemoveTestPath(dir_);
  }

  void TearDown(void) override {
    zvec::test_util::RemoveTestPath(dir_);
  }

  static std::string dir_;
};

std::string HnswStreamerTurboPqTest::dir_("hnsw_streamer_turbo_pq_test_dir/");

// Create and init a quantizer on the raw fp32 meta with the given metric.
static turbo::Quantizer::Pointer make_quantizer(const string &quantizer_name,
                                                 const string &metric_name,
                                                 uint32_t num_chunk) {
  auto quantizer = IndexFactory::CreateQuantizer(quantizer_name);
  EXPECT_TRUE(quantizer != nullptr);
  if (!quantizer) {
    return nullptr;
  }
  IndexMeta meta;
  meta.set_meta(IndexMeta::DataType::DT_FP32, kDim);
  meta.set_metric(metric_name, 0, ailego::Params());
  ailego::Params qparams;
  if (num_chunk > 0) {
    qparams.set("num_chunk", num_chunk);
  }
  EXPECT_EQ(0, quantizer->init(meta, qparams));
  return quantizer;
}

// Build a holder of random fp32 vectors for quantizer training.
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

static void set_hnsw_params(ailego::Params *params) {
  params->set("proxima.hnsw.streamer.max_neighbor_count", 16U);
  params->set("proxima.hnsw.streamer.upper_neighbor_count", 8U);
  params->set("proxima.hnsw.streamer.scaling_factor", 5U);
  params->set("proxima.hnsw.streamer.efconstruction", 100U);
  params->set("proxima.hnsw.streamer.ef", 200U);
  params->set("proxima.hnsw.streamer.get_vector_enable", true);
}

// Recall of hnsw results against brute-force results (same distance basis).
static float calc_recall(const vector<uint64_t> &hnsw_keys,
                         const vector<uint64_t> &bf_keys) {
  std::set<uint64_t> bf_set(bf_keys.begin(), bf_keys.end());
  size_t hit = 0;
  for (auto key : hnsw_keys) {
    if (bf_set.count(key)) {
      ++hit;
    }
  }
  return static_cast<float>(hit) /
         static_cast<float>(std::max<size_t>(1UL, bf_keys.size()));
}

static vector<uint64_t> collect_keys(IndexStreamer::Context::Pointer &ctx) {
  vector<uint64_t> keys;
  auto &result = ctx->result();
  keys.reserve(result.size());
  for (size_t i = 0; i < result.size(); ++i) {
    keys.push_back(result[i].key());
  }
  return keys;
}

//! Build the index with PQ codes (SDC distances), search with an ADC LUT
//! query, compare recall against brute force on the same LUT basis, and
//! verify stored codes via get_vector + dequantize.
static void run_pq_index_test(const string &dir, const string &metric_name,
                              float min_recall) {
  auto quantizer = make_quantizer("PqInt8Quantizer", metric_name, kNumChunk);
  ASSERT_TRUE(quantizer != nullptr);
  ASSERT_EQ(0, quantizer->train(make_train_holder(42)));

  const size_t code_len = quantizer->quantized_datapoint_vector_length();
  const IndexMeta &qz_meta = quantizer->meta();
  ASSERT_EQ(IndexMeta::DataType::DT_INT8, qz_meta.data_type());
  ASSERT_EQ(kNumChunk, qz_meta.dimension());

  // The Cosine metric only supports fp16/fp32; for DT_INT8 storage the
  // existing QuantizedInteger metric carries the origin metric instead.
  IndexMeta stg_meta = qz_meta;
  if (metric_name == "Cosine") {
    ailego::Params mparams;
    mparams.set(QUANTIZED_INTEGER_METRIC_ORIGIN_METRIC_NAME, metric_name);
    stg_meta.set_metric("QuantizedInteger", 0, mparams);
  }

  // Generate raw base vectors and quantize them into PQ codes.
  std::mt19937 gen(7);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  vector<NumericalVector<float>> raw_vecs(kDocCount, NumericalVector<float>(kDim));
  vector<vector<uint8_t>> codes(kDocCount, vector<uint8_t>(code_len));
  for (size_t i = 0; i < kDocCount; ++i) {
    for (size_t j = 0; j < kDim; ++j) {
      raw_vecs[i][j] = dist(gen);
    }
    quantizer->quantize_data(raw_vecs[i].data(), codes[i].data());
  }

  IndexStreamer::Pointer streamer =
      IndexFactory::CreateStreamer("HnswStreamer");
  ASSERT_TRUE(streamer != nullptr);

  ailego::Params params;
  set_hnsw_params(&params);
  ASSERT_EQ(0, streamer->init(stg_meta, params));
  ASSERT_EQ(0, streamer->init_quantizer(quantizer));

  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_TRUE(storage != nullptr);
  ailego::Params stg_params;
  ASSERT_EQ(0, storage->init(stg_params));
  ASSERT_EQ(0, storage->open(dir + "pq_" + metric_name + ".index", true));
  ASSERT_EQ(0, streamer->open(storage));

  auto ctx = streamer->create_context();
  ASSERT_TRUE(!!ctx);

  // qmeta of the stored layout: int8 codes with num_chunk dimensions.
  IndexQueryMeta add_qmeta(IndexMeta::MetaType::MT_DENSE,
                           IndexMeta::DataType::DT_INT8, 1U,
                           static_cast<uint32_t>(kNumChunk),
                           static_cast<uint32_t>(quantizer->type()),
                           qz_meta.extra_meta_size());
  for (size_t i = 0; i < kDocCount; ++i) {
    ASSERT_EQ(0, streamer->add_impl(i, codes[i].data(), add_qmeta, ctx));
  }

  // Search with LUT queries; recall against brute force (same ADC basis).
  IndexQueryMeta raw_qmeta(IndexMeta::DataType::DT_FP32,
                           static_cast<uint32_t>(kDim));
  float recall_sum = 0.0f;
  for (size_t q = 0; q < kQueryCount; ++q) {
    NumericalVector<float> query(kDim);
    for (size_t j = 0; j < kDim; ++j) {
      query[j] = dist(gen);
    }
    std::string lut;
    IndexQueryMeta lut_qmeta;
    ASSERT_EQ(0, quantizer->quantize(query.data(), raw_qmeta, &lut,
                                     &lut_qmeta));
    ASSERT_EQ(quantizer->quantized_query_vector_length(), lut.size());

    ctx->set_topk(kTopk);
    ASSERT_EQ(0, streamer->search_impl(lut.data(), lut_qmeta, ctx));
    vector<uint64_t> hnsw_keys = collect_keys(ctx);
    ASSERT_EQ(kTopk, hnsw_keys.size());

    ASSERT_EQ(0, streamer->search_bf_impl(lut.data(), lut_qmeta, ctx));
    vector<uint64_t> bf_keys = collect_keys(ctx);
    ASSERT_EQ(kTopk, bf_keys.size());

    recall_sum += calc_recall(hnsw_keys, bf_keys);
  }
  float avg_recall = recall_sum / static_cast<float>(kQueryCount);
  EXPECT_GE(avg_recall, min_recall) << "metric=" << metric_name
                                      << " avg_recall=" << avg_recall;

  // Verify stored codes: get_vector + dequantize approximates the raw vector.
  for (size_t i = 0; i < 10; ++i) {
    const void *stored = streamer->get_vector(i);
    ASSERT_TRUE(stored != nullptr);
    std::string recon;
    ASSERT_EQ(0, quantizer->dequantize(stored, add_qmeta, &recon));
    ASSERT_EQ(kDim * sizeof(float), recon.size());
    const float *recon_data = reinterpret_cast<const float *>(recon.data());
    float sq_err = 0.0f;
    float raw_norm = 0.0f;
    for (size_t j = 0; j < kDim; ++j) {
      float diff = recon_data[j] - raw_vecs[i][j];
      sq_err += diff * diff;
      raw_norm += raw_vecs[i][j] * raw_vecs[i][j];
    }
    EXPECT_LT(sq_err, 0.5f * raw_norm) << "i=" << i;
  }

  streamer->flush(0UL);
  streamer.reset();
}

TEST_F(HnswStreamerTurboPqTest, TestPqInt8SquaredEuclidean) {
  run_pq_index_test(dir_, "SquaredEuclidean", 0.8f);
}

TEST_F(HnswStreamerTurboPqTest, TestPqInt8Cosine) {
  run_pq_index_test(dir_, "Cosine", 0.8f);
}

//! Fp32Quantizer is a pass-through quantizer: storage stays fp32 and
//! the quantizer distance path must behave the same as the plain metric
//! path, i.e. hnsw search reaches (nearly) full recall against brute force.
TEST_F(HnswStreamerTurboPqTest, TestFp32QuantizerFallback) {
  auto quantizer = make_quantizer("Fp32Quantizer", "SquaredEuclidean", 0);
  ASSERT_TRUE(quantizer != nullptr);
  const IndexMeta &qz_meta = quantizer->meta();
  ASSERT_EQ(IndexMeta::DataType::DT_FP32, qz_meta.data_type());
  ASSERT_EQ(kDim, qz_meta.dimension());

  std::mt19937 gen(11);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  vector<NumericalVector<float>> raw_vecs(kDocCount,
                                          NumericalVector<float>(kDim));
  for (size_t i = 0; i < kDocCount; ++i) {
    for (size_t j = 0; j < kDim; ++j) {
      raw_vecs[i][j] = dist(gen);
    }
  }

  IndexStreamer::Pointer streamer =
      IndexFactory::CreateStreamer("HnswStreamer");
  ASSERT_TRUE(streamer != nullptr);

  ailego::Params params;
  set_hnsw_params(&params);
  ASSERT_EQ(0, streamer->init(qz_meta, params));
  ASSERT_EQ(0, streamer->init_quantizer(quantizer));

  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_TRUE(storage != nullptr);
  ailego::Params stg_params;
  ASSERT_EQ(0, storage->init(stg_params));
  ASSERT_EQ(0, storage->open(dir_ + "fp32_quantizer.index", true));
  ASSERT_EQ(0, streamer->open(storage));

  auto ctx = streamer->create_context();
  ASSERT_TRUE(!!ctx);

  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32,
                       static_cast<uint32_t>(kDim));
  for (size_t i = 0; i < kDocCount; ++i) {
    ASSERT_EQ(0, streamer->add_impl(i, raw_vecs[i].data(), qmeta, ctx));
  }

  float recall_sum = 0.0f;
  for (size_t q = 0; q < kQueryCount; ++q) {
    NumericalVector<float> query(kDim);
    for (size_t j = 0; j < kDim; ++j) {
      query[j] = dist(gen);
    }
    std::string qz_query;
    IndexQueryMeta qz_qmeta;
    ASSERT_EQ(0, quantizer->quantize(query.data(), qmeta, &qz_query,
                                     &qz_qmeta));
    ASSERT_EQ(kDim * sizeof(float), qz_query.size());

    ctx->set_topk(kTopk);
    ASSERT_EQ(0, streamer->search_impl(qz_query.data(), qz_qmeta, ctx));
    vector<uint64_t> hnsw_keys = collect_keys(ctx);
    ASSERT_EQ(kTopk, hnsw_keys.size());

    ASSERT_EQ(0, streamer->search_bf_impl(qz_query.data(), qz_qmeta, ctx));
    vector<uint64_t> bf_keys = collect_keys(ctx);
    ASSERT_EQ(kTopk, bf_keys.size());

    recall_sum += calc_recall(hnsw_keys, bf_keys);
  }
  float avg_recall = recall_sum / static_cast<float>(kQueryCount);
  EXPECT_GE(avg_recall, 0.95f) << "avg_recall=" << avg_recall;

  // get_vector returns the stored fp32 vector verbatim.
  const void *stored = streamer->get_vector(0);
  ASSERT_TRUE(stored != nullptr);
  EXPECT_EQ(0, std::memcmp(stored, raw_vecs[0].data(), kDim * sizeof(float)));

  streamer->flush(0UL);
  streamer.reset();
}

}  // namespace core
}  // namespace zvec
