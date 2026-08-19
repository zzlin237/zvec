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
#include "hnsw_streamer.h"
#include <iostream>
#include <ailego/internal/cpu_features.h>
#include <ailego/pattern/defer.h>
#include <ailego/utility/memory_helper.h>
#include <zvec/ailego/utility/base64_helper.h>
#include <zvec/core/framework/index_helper.h>
#include "utility/sparse_utility.h"
#include "hnsw_algorithm.h"
#include "hnsw_context.h"
#include "hnsw_dist_calculator.h"
#include "hnsw_index_provider.h"

namespace zvec {
namespace core {

HnswStreamer::HnswStreamer() = default;

HnswStreamer::~HnswStreamer() {
  if (state_ == STATE_INITED || state_ == STATE_OPENED) {
    this->cleanup();
  }
}

int HnswStreamer::init(const IndexMeta &imeta, const ailego::Params &params) {
  meta_ = imeta;
  meta_.set_streamer("HnswStreamer", HnswEntity::kRevision, params);

  params.get(PARAM_HNSW_STREAMER_MAX_INDEX_SIZE, &max_index_size_);

  params.get(PARAM_HNSW_STREAMER_MAX_NEIGHBOR_COUNT, &upper_max_neighbor_cnt_);
  float multiplier = HnswEntity::kDefaultL0MaxNeighborCntMultiplier;
  params.get(PARAM_HNSW_STREAMER_L0_MAX_NEIGHBOR_COUNT_MULTIPLIER, &multiplier);
  l0_max_neighbor_cnt_ = multiplier * upper_max_neighbor_cnt_;

  multiplier = HnswEntity::kDefaultNeighborPruneMultiplier;
  params.get(PARAM_HNSW_STREAMER_NEIGHBOR_PRUNE_MULTIPLIER, &multiplier);
  size_t prune_cnt = multiplier * upper_max_neighbor_cnt_;
  scaling_factor_ = upper_max_neighbor_cnt_;
  params.get(PARAM_HNSW_STREAMER_SCALING_FACTOR, &scaling_factor_);

  params.get(PARAM_HNSW_STREAMER_DOCS_HARD_LIMIT, &docs_hard_limit_);
  params.get(PARAM_HNSW_STREAMER_EF, &ef_);
  params.get(PARAM_HNSW_STREAMER_PO, &po_);
  params.get(PARAM_HNSW_STREAMER_PL, &pl_);
  params.get(PARAM_HNSW_STREAMER_EFCONSTRUCTION, &ef_construction_);
  params.get(PARAM_HNSW_STREAMER_VISIT_BLOOMFILTER_ENABLE, &bf_enabled_);
  params.get(PARAM_HNSW_STREAMER_VISIT_BLOOMFILTER_NEGATIVE_PROB,
             &bf_negative_prob_);
  params.get(PARAM_HNSW_STREAMER_BRUTE_FORCE_THRESHOLD, &bruteforce_threshold_);
  params.get(PARAM_HNSW_STREAMER_MAX_SCAN_RATIO, &max_scan_ratio_);
  params.get(PARAM_HNSW_STREAMER_MAX_SCAN_LIMIT, &max_scan_limit_);
  params.get(PARAM_HNSW_STREAMER_MIN_SCAN_LIMIT, &min_scan_limit_);
  params.get(PARAM_HNSW_STREAMER_CHECK_CRC_ENABLE, &check_crc_enabled_);
  params.get(PARAM_HNSW_STREAMER_CHUNK_SIZE, &chunk_size_);
  params.get(PARAM_HNSW_STREAMER_FILTER_SAME_KEY, &filter_same_key_);
  params.get(PARAM_HNSW_STREAMER_GET_VECTOR_ENABLE, &get_vector_enabled_);
  params.get(PARAM_HNSW_STREAMER_MIN_NEIGHBOR_COUNT, &min_neighbor_cnt_);
  params.get(PARAM_HNSW_STREAMER_FORCE_PADDING_RESULT_ENABLE,
             &force_padding_topk_enabled_);
  params.get(PARAM_HNSW_STREAMER_USE_ID_MAP, &use_id_map_);
  params.get(PARAM_HNSW_STREAMER_USE_CONTIGUOUS_MEMORY,
             &use_contiguous_memory_);
  params.get(PARAM_HNSW_STREAMER_USE_EXTERNAL_VECTOR, &use_external_vector_);

  params.get(PARAM_HNSW_STREAMER_TURBO_QUANTIZER_CLASS,
             &turbo_quantizer_class_);
  params.get(PARAM_HNSW_STREAMER_QG_ENABLE, &qg_enable_);

  params.get(PARAM_HNSW_STREAMER_DOCS_SOFT_LIMIT, &docs_soft_limit_);
  if (docs_soft_limit_ > 0 && docs_soft_limit_ > docs_hard_limit_) {
    LOG_ERROR("[%s] must be >= [%s]",
              PARAM_HNSW_STREAMER_DOCS_HARD_LIMIT.c_str(),
              PARAM_HNSW_STREAMER_DOCS_SOFT_LIMIT.c_str());
    return IndexError_InvalidArgument;
  } else if (docs_soft_limit_ == 0UL) {
    docs_soft_limit_ =
        docs_hard_limit_ * HnswEntity::kDefaultDocsSoftLimitRatio;
  }

  if (ef_ == 0U) {
    ef_ = HnswEntity::kDefaultEf;
  }
  if (ef_construction_ == 0U) {
    ef_construction_ = HnswEntity::kDefaultEfConstruction;
  }
  if (upper_max_neighbor_cnt_ == 0U) {
    upper_max_neighbor_cnt_ = HnswEntity::kDefaultUpperMaxNeighborCnt;
  }
  if (upper_max_neighbor_cnt_ > HnswEntity::kMaxNeighborCnt) {
    LOG_ERROR("[%s] must be in range (0,%d)",
              PARAM_HNSW_STREAMER_MAX_NEIGHBOR_COUNT.c_str(),
              HnswEntity::kMaxNeighborCnt);
    return IndexError_InvalidArgument;
  }
  if (l0_max_neighbor_cnt_ == 0U) {
    l0_max_neighbor_cnt_ = HnswEntity::kDefaultL0MaxNeighborCnt;
  }
  if (l0_max_neighbor_cnt_ > HnswEntity::kMaxNeighborCnt) {
    LOG_ERROR("MaxL0NeighborCnt must be in range (0,%d)",
              HnswEntity::kMaxNeighborCnt);
    return IndexError_InvalidArgument;
  }
  if (min_neighbor_cnt_ > upper_max_neighbor_cnt_) {
    LOG_ERROR("[%s]-[%u] must be <= [%s]-[%u]",
              PARAM_HNSW_STREAMER_MIN_NEIGHBOR_COUNT.c_str(), min_neighbor_cnt_,
              PARAM_HNSW_STREAMER_MAX_NEIGHBOR_COUNT.c_str(),
              upper_max_neighbor_cnt_);
    return IndexError_InvalidArgument;
  }

  if (bf_negative_prob_ <= 0.0f || bf_negative_prob_ >= 1.0f) {
    LOG_ERROR("[%s] must be in range (0,1)",
              PARAM_HNSW_STREAMER_VISIT_BLOOMFILTER_NEGATIVE_PROB.c_str());
    return IndexError_InvalidArgument;
  }

  if (scaling_factor_ == 0U) {
    scaling_factor_ = HnswEntity::kDefaultScalingFactor;
  }
  if (scaling_factor_ < 5 || scaling_factor_ > 1000) {
    LOG_ERROR("[%s] must be in range [5,1000]",
              PARAM_HNSW_STREAMER_SCALING_FACTOR.c_str());
    return IndexError_InvalidArgument;
  }

  if (max_scan_ratio_ <= 0.0f || max_scan_ratio_ > 1.0f) {
    LOG_ERROR("[%s] must be in range (0.0f,1.0f]",
              PARAM_HNSW_STREAMER_MAX_SCAN_RATIO.c_str());
    return IndexError_InvalidArgument;
  }

  if (max_scan_limit_ < min_scan_limit_) {
    LOG_ERROR("[%s] must be >= [%s]",
              PARAM_HNSW_STREAMER_MAX_SCAN_LIMIT.c_str(),
              PARAM_HNSW_STREAMER_MIN_SCAN_LIMIT.c_str());
    return IndexError_InvalidArgument;
  }

  if (prune_cnt == 0UL) {
    prune_cnt = upper_max_neighbor_cnt_;
  }
  prune_cnt_ = prune_cnt;
  if (chunk_size_ == 0UL) {
    chunk_size_ = HnswEntity::kDefaultChunkSize;
  }
  if (chunk_size_ > HnswEntity::kMaxChunkSize) {
    LOG_ERROR("[%s] must be < %zu", PARAM_HNSW_STREAMER_CHUNK_SIZE.c_str(),
              HnswEntity::kMaxChunkSize);
    return IndexError_InvalidArgument;
  }

  LOG_DEBUG(
      "Init params: maxIndexSize=%zu docsHardLimit=%zu docsSoftLimit=%zu "
      "efConstruction=%u ef=%u upperMaxNeighborCnt=%u l0MaxNeighborCnt=%u "
      "scalingFactor=%u maxScanRatio=%.3f minScanLimit=%zu maxScanLimit=%zu "
      "bfEnabled=%d bruteFoceThreshold=%zu bfNegativeProbability=%.5f "
      "checkCrcEnabled=%d pruneSize=%u vectorSize=%u chunkSize=%zu "
      "filterSameKey=%u getVectorEnabled=%u minNeighborCount=%u "
      "forcePadding=%u ",
      max_index_size_, docs_hard_limit_, docs_soft_limit_, ef_construction_,
      ef_, upper_max_neighbor_cnt_, l0_max_neighbor_cnt_, scaling_factor_,
      max_scan_ratio_, min_scan_limit_, max_scan_limit_, bf_enabled_,
      bruteforce_threshold_, bf_negative_prob_, check_crc_enabled_, prune_cnt_,
      meta_.element_size(), chunk_size_, filter_same_key_, get_vector_enabled_,
      min_neighbor_cnt_, force_padding_topk_enabled_);

  state_ = STATE_INITED;

  return 0;
}

int HnswStreamer::cleanup(void) {
  if (state_ == STATE_OPENED) {
    this->close();
  }

  LOG_INFO("HnswStreamer cleanup");

  meta_.clear();
  metric_.reset();
  add_quantizer_.reset();
  search_quantizer_.reset();
  turbo_quantizer_class_.clear();
  stats_.clear();
  provider_.reset();
  provider_meta_.clear();
  provider_metric_.reset();
  if (entity_) {
    entity_->cleanup();
  }

  if (alg_) {
    alg_->cleanup();
  }

  max_index_size_ = 0UL;
  docs_hard_limit_ = HnswEntity::kDefaultDocsHardLimit;
  docs_soft_limit_ = 0UL;
  upper_max_neighbor_cnt_ = HnswEntity::kDefaultUpperMaxNeighborCnt;
  l0_max_neighbor_cnt_ = HnswEntity::kDefaultL0MaxNeighborCnt;
  ef_ = HnswEntity::kDefaultEf;
  ef_construction_ = HnswEntity::kDefaultEfConstruction;
  bf_enabled_ = false;
  scaling_factor_ = HnswEntity::kDefaultScalingFactor;
  bruteforce_threshold_ = HnswEntity::kDefaultBruteForceThreshold;
  max_scan_limit_ = HnswEntity::kDefaultMaxScanLimit;
  min_scan_limit_ = HnswEntity::kDefaultMinScanLimit;
  chunk_size_ = HnswEntity::kDefaultChunkSize;
  bf_negative_prob_ = HnswEntity::kDefaultBFNegativeProbability;
  max_scan_ratio_ = HnswEntity::kDefaultScanRatio;
  state_ = STATE_INIT;
  check_crc_enabled_ = false;
  filter_same_key_ = false;
  get_vector_enabled_ = false;

  return 0;
}

zvec::turbo::Quantizer::Pointer HnswStreamer::create_turbo_quantizer(
    const std::string &class_name) const {
  auto quantizer = IndexFactory::CreateQuantizer(class_name);
  if (!quantizer) {
    LOG_WARN("HnswStreamer: failed to create quantizer '%s'",
             class_name.c_str());
    return nullptr;
  }

  ailego::Params quantizer_params;
  auto &sp = meta_.streamer_params();
  int nsq = 0;
  if (sp.get("num_chunk", &nsq)) {
    quantizer_params.set("num_chunk", nsq);
  }
  bool use_zero_mean = false;
  if (sp.get("use_zero_mean", &use_zero_mean)) {
    quantizer_params.set("use_zero_mean", use_zero_mean);
  }
  // Optional OPQ rotation (build-time only; the trained matrix travels with
  // the serialized quantizer, so the reopen path does not depend on these).
  std::string rotate_type;
  if (sp.get("rotate_type", &rotate_type)) {
    quantizer_params.set("rotate_type", rotate_type);
  }
  uint32_t opq_iter = 0;
  if (sp.get("opq_iter", &opq_iter)) {
    quantizer_params.set("opq_iter", opq_iter);
  }
  uint32_t opq_pq_iter = 0;
  if (sp.get("opq_pq_iter", &opq_pq_iter)) {
    quantizer_params.set("opq_pq_iter", opq_pq_iter);
  }

  int ret = quantizer->init(meta_, quantizer_params);
  if (ret != 0) {
    LOG_ERROR("Failed to init turbo quantizer '%s', ret=%d", class_name.c_str(),
              ret);
    return nullptr;
  }
  return quantizer;
}

int HnswStreamer::setup_entity() {
  entity_->set_use_key_info_map(use_id_map_);
  entity_->set_ef_construction(ef_construction_);
  entity_->set_upper_neighbor_cnt(upper_max_neighbor_cnt_);
  entity_->set_l0_neighbor_cnt(l0_max_neighbor_cnt_);
  entity_->set_scaling_factor(scaling_factor_);
  entity_->set_prune_cnt(prune_cnt_);
  // Determine the per-node vector storage size.
  // Priority: persisted value from a previous build > live quantizer > meta.
  // The persisted value (entity_vector_size) bridges the gap between the
  // external IndexMeta contract (raw FP32 element_size) and the actual
  // internal storage format (e.g. PQ codes) when no converter/reformer
  // layer is present. For external-vector entities the per-node vector
  // prefix is removed; vector_size stays 0 and the distance dimension is
  // taken from meta.dimension().
  size_t vec_size = 0;
  auto &sp = meta_.streamer_params();
  uint64_t persisted_vs = 0;
  if (sp.get("entity_vector_size", &persisted_vs) && persisted_vs > 0) {
    vec_size = static_cast<size_t>(persisted_vs);
  } else if (!use_external_vector_ && add_quantizer_) {
    vec_size = add_quantizer_->quantized_datapoint_vector_length();
  } else if (use_external_vector_) {
    vec_size = 0;
  } else {
    vec_size = meta_.element_size();
  }
  // Persist so the next open() can read it before the quantizer is restored.
  meta_.mutable_streamer_params()->set("entity_vector_size",
                                       static_cast<uint64_t>(vec_size));
  entity_->set_vector_size(vec_size);

  // Quantized graph region (see hnsw_qg.h).  Same bridging problem as the
  // vector size: the geometry decides node_size, so it must be known before
  // entity init() and therefore before the quantizer is restored.  Priority:
  // persisted values from a previous build > live quantizer capability.
  uint64_t qg_block_vectors = 0;
  uint64_t qg_block_bytes = 0;
  bool qg_materialized = false;
  if (sp.get("entity_qg_block_bytes", &qg_block_bytes) &&
      sp.get("entity_qg_block_vectors", &qg_block_vectors)) {
    sp.get("entity_qg_materialized", &qg_materialized);
  } else if (qg_enable_) {
    //! The geometry only depends on the code layout, not on trained data, so
    //! when the real quantizer does not exist yet (a new index creates it
    //! after this point) an untrained probe instance answers just as well.
    auto source = add_quantizer_;
    if (!source && !turbo_quantizer_class_.empty()) {
      source = create_turbo_quantizer(turbo_quantizer_class_);
    }
    auto packer =
        std::dynamic_pointer_cast<const zvec::turbo::PackedCodeQuantizer>(
            source);
    if (packer) {
      qg_block_vectors = packer->packed_block_vectors();
      qg_block_bytes = packer->packed_block_bytes();
    } else {
      LOG_WARN(
          "Quantizer '%s' has no packed-code capability, "
          "quantized graph disabled",
          turbo_quantizer_class_.c_str());
    }
  }
  if (qg_block_bytes > 0 && qg_block_vectors > 0) {
    entity_->set_qg_layout(static_cast<size_t>(qg_block_vectors),
                           static_cast<size_t>(qg_block_bytes));
    entity_->set_qg_materialized(qg_materialized);
    meta_.mutable_streamer_params()->set("entity_qg_block_vectors",
                                         qg_block_vectors);
    meta_.mutable_streamer_params()->set("entity_qg_block_bytes",
                                         qg_block_bytes);
    meta_.mutable_streamer_params()->set("entity_qg_materialized",
                                         qg_materialized);
  }

  entity_->set_chunk_size(chunk_size_);
  entity_->set_filter_same_key(filter_same_key_);
  entity_->set_get_vector(get_vector_enabled_);
  entity_->set_min_neighbor_cnt(min_neighbor_cnt_);

  int ret = entity_->init(docs_hard_limit_);
  if (ret != 0) {
    LOG_ERROR("Hnsw entity init failed for %s", IndexError::What(ret));
  }
  return ret;
}

static std::string QuantizerClassName(
    const zvec::turbo::Quantizer::Pointer &q) {
  if (!q) return {};
  //! Only types that map to exactly one registered class can be recovered
  //! this way: kPQ is shared by PqInt8Quantizer / PqInt4Quantizer and kRecord
  //! by Int8Quantizer / Int4Quantizer, so those must be told apart by the
  //! turbo_quantizer_class parameter instead.
  switch (q->type()) {
    case zvec::turbo::QuantizeType::kFp32:
      return "Fp32Quantizer";
    case zvec::turbo::QuantizeType::kFp16:
      return "Fp16Quantizer";
    case zvec::turbo::QuantizeType::kPQFast:
      return "PqFastQuantizer";
    default:
      return {};
  }
}

int HnswStreamer::init_quantizer(zvec::turbo::Quantizer::Pointer quantizer) {
  add_quantizer_ = quantizer;
  search_quantizer_ = quantizer;
  if (turbo_quantizer_class_.empty()) {
    turbo_quantizer_class_ = QuantizerClassName(quantizer);
  }

  return 0;
}

int HnswStreamer::init_quantizer(
    zvec::turbo::Quantizer::Pointer add_quantizer,
    zvec::turbo::Quantizer::Pointer search_quantizer) {
  add_quantizer_ = add_quantizer;
  search_quantizer_ = search_quantizer;
  if (turbo_quantizer_class_.empty()) {
    turbo_quantizer_class_ = QuantizerClassName(add_quantizer);
  }

  return 0;
}

int HnswStreamer::open(IndexStorage::Pointer stg) {
  LOG_INFO("HnswStreamer open");

  if (ailego_unlikely(state_ != STATE_INITED)) {
    LOG_ERROR("Open storage failed, init streamer first!");
    return IndexError_NoReady;
  }

  // Create entity based on storage type
  switch (stg->memory_block_type()) {
    case IndexStorage::MemoryBlock::MBT_BUFFERPOOL: {
      entity_ = std::make_unique<HnswBufferPoolStreamerEntity>(stats_);
      break;
    }
    default: {
      if (use_external_vector_) {
        entity_ = std::make_unique<HnswExternalStreamerEntity>(stats_);
      } else if (use_contiguous_memory_) {
        entity_ = std::make_unique<HnswContiguousStreamerEntity>(stats_);
      } else {
        entity_ = std::make_unique<HnswMmapStreamerEntity>(stats_);
      }
      break;
    }
  }
  // For an existing index, read the persisted entity_vector_size from
  // IndexMeta BEFORE setup_entity() so the correct internal storage
  // vector size is used from the start (even when the turbo quantizer
  // has not been restored yet).
  {
    IndexMeta stored_meta;
    int meta_ret = IndexHelper::DeserializeFromStorage(stg.get(), &stored_meta);
    if (meta_ret == 0 && !stored_meta.streamer_name().empty()) {
      uint64_t persisted_vs = 0;
      if (stored_meta.streamer_params().get("entity_vector_size",
                                            &persisted_vs) &&
          persisted_vs > 0) {
        meta_.mutable_streamer_params()->set("entity_vector_size",
                                             persisted_vs);
      }
      //! Same reason for the quantized graph geometry: it decides node_size,
      //! which setup_entity() must compute exactly as the previous build did.
      uint64_t qg_bytes = 0;
      uint64_t qg_vectors = 0;
      if (stored_meta.streamer_params().get("entity_qg_block_bytes",
                                            &qg_bytes) &&
          stored_meta.streamer_params().get("entity_qg_block_vectors",
                                            &qg_vectors)) {
        bool qg_materialized = false;
        stored_meta.streamer_params().get("entity_qg_materialized",
                                          &qg_materialized);
        meta_.mutable_streamer_params()->set("entity_qg_block_bytes", qg_bytes);
        meta_.mutable_streamer_params()->set("entity_qg_block_vectors",
                                             qg_vectors);
        meta_.mutable_streamer_params()->set("entity_qg_materialized",
                                             qg_materialized);
      }
    }
  }
  int ret = setup_entity();
  if (ret != 0) {
    return ret;
  }

  ret = entity_->open(std::move(stg), max_index_size_, check_crc_enabled_);
  if (ret != 0) {
    return ret;
  }
  IndexMeta index_meta;
  ret = entity_->get_index_meta(&index_meta);
  if (ret == IndexError_NoExist) {
    // New index: defer writing meta to storage until close(), when the
    // quantizer data has been serialized into meta_.  Writing a small meta
    // here would allocate a small segment that cannot hold the much larger
    // meta at close time (with base64 PQ codebook).  The in-memory meta_ is
    // sufficient for the entity during build.
  } else if (ret != 0) {
    LOG_ERROR("Failed to get index meta for %s", IndexError::What(ret));
    return ret;
  } else {
    if (index_meta.dimension() != meta_.dimension() ||
        index_meta.element_size() != meta_.element_size() ||
        index_meta.metric_name() != meta_.metric_name() ||
        index_meta.data_type() != meta_.data_type()) {
      LOG_ERROR("IndexMeta mismatch from the previous in index");
      return IndexError_Mismatch;
    }
    // The IndexMetric Params may be updated like MipsSquaredEuclidean
    auto metric_params = index_meta.metric_params();
    metric_params.merge(meta_.metric_params());
    meta_.set_metric(index_meta.metric_name(), 0, metric_params);

    // Restore converter/reformer from the persisted meta
    if (!index_meta.reformer_name().empty()) {
      meta_.set_reformer(index_meta.reformer_name(),
                         index_meta.reformer_revision(),
                         index_meta.reformer_params());
    }
    if (!index_meta.converter_name().empty()) {
      meta_.set_converter(index_meta.converter_name(),
                          index_meta.converter_revision(),
                          index_meta.converter_params());
    }
    // Restore streamer_params (contains persisted quantizer data, etc.)
    if (!index_meta.streamer_name().empty()) {
      meta_.set_streamer(index_meta.streamer_name(),
                         index_meta.streamer_revision(),
                         index_meta.streamer_params());
    }
  }

  metric_ = IndexFactory::CreateMetric(meta_.metric_name());
  if (!metric_) {
    LOG_ERROR("Failed to create metric %s", meta_.metric_name().c_str());
    return IndexError_NoExist;
  }
  ret = metric_->init(meta_, meta_.metric_params());
  if (ret != 0) {
    LOG_ERROR("Failed to init metric, ret=%d", ret);
    return ret;
  }

  if (!metric_->distance()) {
    LOG_ERROR("Invalid metric distance");
    return IndexError_InvalidArgument;
  }

  if (!metric_->batch_distance()) {
    LOG_ERROR("Invalid metric batch distance");
    return IndexError_InvalidArgument;
  }

  add_distance_ = metric_->distance();
  add_batch_distance_ = metric_->batch_distance();

  search_distance_ = add_distance_;
  search_batch_distance_ = add_batch_distance_;

  if (metric_->query_metric() && metric_->query_metric()->distance() &&
      metric_->query_metric()->batch_distance()) {
    search_distance_ = metric_->query_metric()->distance();
    search_batch_distance_ = metric_->query_metric()->batch_distance();
  }

  //! Create a dedicated build metric when the provider meta differs from
  //! the index meta in layout or metric, so build distances run in the
  //! original vector space
  if (provider_) {
    const bool layout_differs =
        provider_meta_.data_type() != meta_.data_type() ||
        provider_meta_.dimension() != meta_.dimension() ||
        provider_meta_.element_size() != meta_.element_size();
    const bool use_index_metric = provider_meta_.metric_name().empty();
    if (layout_differs || !use_index_metric) {
      const std::string &metric_name =
          use_index_metric ? meta_.metric_name() : provider_meta_.metric_name();
      const ailego::Params &metric_params =
          use_index_metric ? meta_.metric_params()
                           : provider_meta_.metric_params();
      provider_metric_ = IndexFactory::CreateMetric(metric_name);
      if (!provider_metric_) {
        LOG_ERROR("Failed to create provider metric %s", metric_name.c_str());
        return IndexError_NoExist;
      }
      ret = provider_metric_->init(provider_meta_, metric_params);
      if (ret != 0) {
        LOG_ERROR("Failed to init provider metric, ret=%d", ret);
        return ret;
      }
      if (!provider_metric_->distance() ||
          !provider_metric_->batch_distance()) {
        LOG_ERROR("Invalid provider metric distance");
        return IndexError_InvalidArgument;
      }
      add_distance_ = provider_metric_->distance();
      add_batch_distance_ = provider_metric_->batch_distance();
    }
  }

  // Restore turbo quantizer from persisted IndexMeta (existing index),
  // or auto-create a fresh one for new indexes.
  if (!add_quantizer_) {
    std::string quantizer_class;
    std::string quantizer_data_b64;
    auto &sp = meta_.streamer_params();
    bool has_class =
        sp.get(PARAM_HNSW_STREAMER_TURBO_QUANTIZER_CLASS, &quantizer_class);
    bool has_data = sp.get("turbo_quantizer_data_b64", &quantizer_data_b64);
    bool has_persisted = has_class && has_data;

    if (has_persisted && !quantizer_class.empty() &&
        !quantizer_data_b64.empty()) {
      // Base64-decode the binary quantizer data.
      std::string quantizer_data =
          ailego::Base64Helper::Decode(quantizer_data_b64);
      // Restore quantizer from serialized state in IndexMeta.
      add_quantizer_ = IndexFactory::CreateQuantizer(quantizer_class);
      if (add_quantizer_) {
        // Init BEFORE deserialize so the metric context (normalization,
        // extra-meta size, distance function) flows from the current meta_
        // exactly as it does for a fresh index.  The construction params
        // (num_chunk, use_zero_mean) were persisted in streamer_params at
        // build time and restored above, so init() can reconstruct the same
        // configuration; deserialize() then loads the codebook on top.
        ailego::Params quantizer_params;
        int nsq = 0;
        if (sp.get("num_chunk", &nsq)) {
          quantizer_params.set("num_chunk", nsq);
        }
        bool use_zero_mean = false;
        if (sp.get("use_zero_mean", &use_zero_mean)) {
          quantizer_params.set("use_zero_mean", use_zero_mean);
        }
        std::string rotate_type;
        if (sp.get("rotate_type", &rotate_type)) {
          quantizer_params.set("rotate_type", rotate_type);
        }
        ret = add_quantizer_->init(meta_, quantizer_params);
        if (ret != 0) {
          LOG_ERROR(
              "Failed to init turbo quantizer '%s' before restore, "
              "ret=%d",
              quantizer_class.c_str(), ret);
          add_quantizer_.reset();
        } else {
          ret = add_quantizer_->deserialize(quantizer_data);
          if (ret != 0) {
            LOG_ERROR("Failed to deserialize turbo quantizer '%s', ret=%d",
                      quantizer_class.c_str(), ret);
            add_quantizer_.reset();
          } else {
            turbo_quantizer_class_ = quantizer_class;
            search_quantizer_ = add_quantizer_;
            LOG_INFO("HnswStreamer: restored turbo quantizer '%s' from index",
                     quantizer_class.c_str());
          }
        }
      }
    } else if (!turbo_quantizer_class_.empty()) {
      // New index: create and init a fresh quantizer.
      add_quantizer_ = create_turbo_quantizer(turbo_quantizer_class_);
      if (add_quantizer_) {
        search_quantizer_ = add_quantizer_;
        LOG_INFO("HnswStreamer: using turbo quantizer '%s'",
                 turbo_quantizer_class_.c_str());
      } else {
        LOG_WARN(
            "HnswStreamer: quantizer '%s' unavailable, "
            "falling back to metric distance",
            turbo_quantizer_class_.c_str());
      }
    } else {
      LOG_INFO(
          "HnswStreamer: no turbo quantizer configured, "
          "using legacy metric distance path");
    }
  }

  // Create algorithm based on entity storage mode
  switch (entity_->storage_mode()) {
    case HnswStorageMode::kBufferPool:
      alg_ = HnswAlgorithmBase::UPointer(
          new HnswAlgorithm<HnswBufferPoolStreamerEntity>(
              static_cast<HnswBufferPoolStreamerEntity &>(*entity_)));
      break;
    case HnswStorageMode::kContiguous: {
      auto &contiguous_entity =
          static_cast<HnswContiguousStreamerEntity &>(*entity_);
      int build_ret = contiguous_entity.build_contiguous_memory();
      if (build_ret != 0) {
        LOG_ERROR("Failed to build contiguous memory, ret=%d", build_ret);
        return build_ret;
      }
      alg_ = HnswAlgorithmBase::UPointer(
          new HnswAlgorithm<HnswContiguousStreamerEntity>(contiguous_entity));
      break;
    }
    case HnswStorageMode::kExternal:
      alg_ = HnswAlgorithmBase::UPointer(
          new HnswAlgorithm<HnswExternalStreamerEntity>(
              static_cast<HnswExternalStreamerEntity &>(*entity_)));
      break;
    default:
      alg_ =
          HnswAlgorithmBase::UPointer(new HnswAlgorithm<HnswMmapStreamerEntity>(
              static_cast<HnswMmapStreamerEntity &>(*entity_)));
      break;
  }
  ret = alg_->init();
  if (ret != 0) {
    return ret;
  }

  state_ = STATE_OPENED;
  magic_ = IndexContext::GenerateMagic();

  return 0;
}

void HnswStreamer::persist_quantizer_to_meta() {
  if (!add_quantizer_) {
    return;
  }
  std::string quantizer_data;
  int qret = add_quantizer_->serialize(&quantizer_data);
  if (qret == 0) {
    // Base64-encode binary data so it survives JSON serialization in Params.
    std::string encoded = ailego::Base64Helper::Encode(quantizer_data.data(),
                                                       quantizer_data.size());
    meta_.mutable_streamer_params()->set(
        PARAM_HNSW_STREAMER_TURBO_QUANTIZER_CLASS, turbo_quantizer_class_);
    meta_.mutable_streamer_params()->set("turbo_quantizer_data_b64",
                                         std::move(encoded));
  } else {
    LOG_ERROR("Failed to serialize turbo quantizer, ret=%d", qret);
  }
}

int HnswStreamer::invalidate_qg() {
  const std::lock_guard<std::mutex> lk(mutex_);
  //! Re-check under the lock: concurrent adds all reach this point, only the
  //! first one has to do the work.
  if (!entity_->qg_ready()) {
    return 0;
  }

  int ret = entity_->invalidate_qg();
  if (ret != 0) {
    LOG_ERROR("Failed to invalidate the quantized graph, ret=%d", ret);
    return ret;
  }
  meta_.mutable_streamer_params()->set("entity_qg_materialized", false);

  //! Search contexts hold a clone of the entity whose header snapshot still
  //! says "materialized"; without a new magic they would keep scanning the now
  //! stale region.  Bumping it forces every context to re-clone on next use.
  magic_ = IndexContext::GenerateMagic();
  LOG_INFO("Quantized graph invalidated by an insert, rebuilt on next flush");
  return 0;
}

int HnswStreamer::materialize_qg_if_needed() {
  if (!entity_->qg_enabled() || entity_->qg_ready()) {
    return 0;
  }

  auto packer =
      std::dynamic_pointer_cast<const zvec::turbo::PackedCodeQuantizer>(
          add_quantizer_);
  if (!packer) {
    LOG_ERROR(
        "Quantized graph region is reserved but quantizer '%s' exposes no "
        "packed-code capability",
        turbo_quantizer_class_.c_str());
    return IndexError_Unsupported;
  }

  //! Exclusive: materialization reads every neighbor list, so no vector may
  //! be added while it runs.  An insert afterwards invalidates the region and
  //! the next flush rebuilds it.
  shared_mutex_.lock();
  AILEGO_DEFER([&]() { shared_mutex_.unlock(); });

  int ret = entity_->materialize_qg(packer.get(), 0U);
  if (ret != 0) {
    LOG_ERROR("Failed to materialize the quantized graph, ret=%d", ret);
    return ret;
  }
  meta_.mutable_streamer_params()->set("entity_qg_materialized", true);

  //! Contexts hold a clone of the entity, whose header snapshot still says
  //! "not materialized".  Invalidating the magic makes every context re-clone
  //! on next use, so searches pick up the region.
  magic_ = IndexContext::GenerateMagic();
  return 0;
}

int HnswStreamer::close(void) {
  LOG_INFO("HnswStreamer close");

  int qg_ret = materialize_qg_if_needed();
  if (qg_ret != 0) {
    return qg_ret;
  }

  stats_.clear();
  meta_.set_metric(metric_->name(), 0, metric_->params());
  persist_quantizer_to_meta();
  entity_->set_index_meta(meta_);
  int ret = entity_->close();
  if (ret != 0) {
    return ret;
  }
  state_ = STATE_INITED;

  return 0;
}

int HnswStreamer::flush(uint64_t checkpoint) {
  LOG_INFO("HnswStreamer flush checkpoint=%zu", (size_t)checkpoint);

  int qg_ret = materialize_qg_if_needed();
  if (qg_ret != 0) {
    return qg_ret;
  }

  meta_.set_metric(metric_->name(), 0, metric_->params());
  persist_quantizer_to_meta();
  entity_->set_index_meta(meta_);
  return entity_->flush(checkpoint);
}

int HnswStreamer::dump(const IndexDumper::Pointer &dumper) {
  LOG_INFO("HnswStreamer dump");

  shared_mutex_.lock();
  AILEGO_DEFER([&]() { shared_mutex_.unlock(); });

  persist_quantizer_to_meta();

  int ret = IndexHelper::SerializeToDumper(meta_, dumper.get());
  if (ret != 0) {
    LOG_ERROR("Failed to serialize meta into dumper.");
    return ret;
  }
  return entity_->dump(dumper);
}

IndexStreamer::Context::Pointer HnswStreamer::create_context(void) const {
  if (ailego_unlikely(state_ != STATE_OPENED)) {
    LOG_ERROR("Create context failed, open storage first!");
    return Context::Pointer();
  }

  HnswEntity::Pointer entity = entity_->clone();
  if (ailego_unlikely(!entity)) {
    LOG_ERROR("CreateContext clone init failed");
    return Context::Pointer();
  }
  HnswContext *ctx = new (std::nothrow)
      HnswContext(meta_.dimension(), metric_, entity, search_quantizer_);
  if (ailego_unlikely(ctx == nullptr)) {
    LOG_ERROR("Failed to new HnswContext");
    return Context::Pointer();
  }
  ctx->set_ef(ef_);
  ctx->set_po(po_);
  ctx->set_pl(pl_);
  ctx->set_max_scan_limit(max_scan_limit_);
  ctx->set_min_scan_limit(min_scan_limit_);
  ctx->set_max_scan_ratio(max_scan_ratio_);
  ctx->set_filter_mode(bf_enabled_ ? VisitFilter::BloomFilter
                                   : VisitFilter::ByteMap);
  ctx->set_filter_negative_probability(bf_negative_prob_);
  ctx->set_magic(magic_);
  ctx->set_force_padding_topk(force_padding_topk_enabled_);
  ctx->set_bruteforce_threshold(bruteforce_threshold_);

  if (ailego_unlikely(ctx->init(HnswContext::kStreamerContext)) != 0) {
    LOG_ERROR("Init HnswContext failed");
    delete ctx;
    return Context::Pointer();
  }
  uint32_t estimate_doc_count = 0;
  if (meta_.streamer_params().get(PARAM_HNSW_STREAMER_ESTIMATE_DOC_COUNT,
                                  &estimate_doc_count)) {
    LOG_DEBUG("HnswStreamer doc_count[%zu] estimate[%zu]",
              (size_t)entity_->doc_cnt(), (size_t)estimate_doc_count);
  }
  ctx->check_need_adjuct_ctx(std::max(entity_->doc_cnt(), estimate_doc_count));

  return Context::Pointer(ctx);
}

IndexProvider::Pointer HnswStreamer::create_provider(void) const {
  LOG_DEBUG("HnswStreamer create provider");

  auto entity = entity_->clone();
  if (ailego_unlikely(!entity)) {
    LOG_ERROR("Clone HnswEntity failed");
    return nullptr;
  }
  return Provider::Pointer(
      new HnswIndexProvider(meta_, entity, "HnswStreamer"));
}

int HnswStreamer::update_context(HnswContext *ctx) const {
  const HnswEntity::Pointer entity = entity_->clone();
  if (!entity) {
    LOG_ERROR("Failed to clone search context entity");
    return IndexError_Runtime;
  }
  ctx->set_max_scan_limit(max_scan_limit_);
  ctx->set_min_scan_limit(min_scan_limit_);
  ctx->set_max_scan_ratio(max_scan_ratio_);
  ctx->set_bruteforce_threshold(bruteforce_threshold_);
  return ctx->update_context(HnswContext::kStreamerContext, meta_, metric_,
                             entity, magic_, search_quantizer_);
}

//! Add a vector with id into index
int HnswStreamer::add_with_id_impl(uint32_t id, const void *query,
                                   const IndexQueryMeta &qmeta,
                                   IndexStreamer::Context::Pointer &context) {
  int ret = check_params(query, qmeta);
  if (ailego_unlikely(ret != 0)) {
    return ret;
  }

  HnswContext *ctx = dynamic_cast<HnswContext *>(context.get());
  ailego_do_if_false(ctx) {
    LOG_ERROR("Cast context to HnswContext failed");
    return IndexError_Cast;
  }
  if (ctx->magic() != magic_) {
    //! context is created by another searcher or streamer
    ret = update_context(ctx);
    if (ret != 0) {
      return ret;
    }
  }

  if (ailego_unlikely(entity_->doc_cnt() >= docs_soft_limit_)) {
    if (entity_->doc_cnt() >= docs_hard_limit_) {
      LOG_ERROR("Current docs %u exceed [%s]", entity_->doc_cnt(),
                PARAM_HNSW_STREAMER_DOCS_HARD_LIMIT.c_str());
      const std::lock_guard<std::mutex> lk(mutex_);
      (*stats_.mutable_discarded_count())++;
      return IndexError_IndexFull;
    } else {
      LOG_WARN("Current docs %u exceed [%s]", entity_->doc_cnt(),
               PARAM_HNSW_STREAMER_DOCS_SOFT_LIMIT.c_str());
    }
  }
  if (ailego_unlikely(!shared_mutex_.try_lock_shared())) {
    LOG_ERROR("Cannot add vector while dumping index");
    (*stats_.mutable_discarded_count())++;
    return IndexError_Unsupported;
  }
  AILEGO_DEFER([&]() { shared_mutex_.unlock_shared(); });

  //! The quantized graph region holds each node's neighbor codes, so this
  //! insert invalidates it: the next flush rebuilds the region, and until then
  //! searches fall back to the per-candidate path.
  if (ailego_unlikely(entity_->qg_ready())) {
    ret = invalidate_qg();
    if (ailego_unlikely(ret != 0)) {
      (*stats_.mutable_discarded_count())++;
      return ret;
    }
  }

  ctx->clear();
  ctx->bind_dist_space(add_distance_, add_batch_distance_, provider_);
  ctx->update_dist_caculator_quantizer(add_quantizer_, /*symmetric=*/true);
  ctx->check_need_adjuct_ctx(entity_->doc_cnt());

  //! use the original vector from provider as the build query, fetched
  //! before mutating the entity so a missing vector cannot leave an
  //! orphan node in the index; a node added with id is keyed by id
  IndexStorage::MemoryBlock original_query_block;
  if (ailego_unlikely(provider_ != nullptr)) {
    ret = provider_->get_vector(id, original_query_block);
    if (ailego_unlikely(ret != 0)) {
      LOG_ERROR("Failed to get original vector from provider, id=%u", id);
      (*stats_.mutable_discarded_count())++;
      return ret;
    }
    ctx->reset_query_raw(original_query_block.data(), provider_meta_);
  } else {
    ctx->reset_query(query, meta_);
  }

  if (metric_->support_train()) {
    const std::lock_guard<std::mutex> lk(mutex_);
    ret = metric_->train(query, meta_.dimension());
    if (ailego_unlikely(ret != 0)) {
      LOG_ERROR("Hnsw streamer metric train failed");
      (*stats_.mutable_discarded_count())++;
      return ret;
    }
  }

  // Encode the raw vector into a PQ code for storage and for symmetric
  // (code-vs-code) distance computation during graph construction when
  // the turbo quantizer works internally.
  const void *store_data = query;
  std::string pq_code_buf;
  if (use_internal_quantizer(add_quantizer_, qmeta)) {
    pq_code_buf.resize(add_quantizer_->quantized_datapoint_vector_length());
    add_quantizer_->quantize_data(query, &pq_code_buf[0]);
    store_data = pq_code_buf.data();
  }
  // With a provider the build query is the original provider vector bound
  // above; only the non-provider path rebinds it to the stored data.
  if (provider_ == nullptr) {
    ctx->reset_query(store_data, meta_);
  }

  level_t level = alg_->get_random_level();
  ret = entity_->add_vector_with_id(level, id, store_data);
  if (ailego_unlikely(ret != 0)) {
    LOG_ERROR("Hnsw streamer add vector failed");
    (*stats_.mutable_discarded_count())++;
    return ret;
  }

  ret = alg_->add_node(id, level, ctx);
  if (ailego_unlikely(ret != 0)) {
    LOG_ERROR("Hnsw steamer add node failed");
    (*stats_.mutable_discarded_count())++;
    return ret;
  }

  if (ailego_unlikely(ctx->error())) {
    (*stats_.mutable_discarded_count())++;
    return IndexError_Runtime;
  }
  (*stats_.mutable_added_count())++;

  return 0;
}

//! Add a vector into index
int HnswStreamer::add_impl(uint64_t pkey, const void *query,
                           const IndexQueryMeta &qmeta,
                           IndexStreamer::Context::Pointer &context) {
  int ret = check_params(query, qmeta);
  if (ailego_unlikely(ret != 0)) {
    return ret;
  }

  HnswContext *ctx = dynamic_cast<HnswContext *>(context.get());
  ailego_do_if_false(ctx) {
    LOG_ERROR("Cast context to HnswContext failed");
    return IndexError_Cast;
  }
  if (ctx->magic() != magic_) {
    //! context is created by another searcher or streamer
    ret = update_context(ctx);
    if (ret != 0) {
      return ret;
    }
  }

  if (ailego_unlikely(entity_->doc_cnt() >= docs_soft_limit_)) {
    if (entity_->doc_cnt() >= docs_hard_limit_) {
      LOG_ERROR("Current docs %u exceed [%s]", entity_->doc_cnt(),
                PARAM_HNSW_STREAMER_DOCS_HARD_LIMIT.c_str());
      const std::lock_guard<std::mutex> lk(mutex_);
      (*stats_.mutable_discarded_count())++;
      return IndexError_IndexFull;
    } else {
      LOG_WARN("Current docs %u exceed [%s]", entity_->doc_cnt(),
               PARAM_HNSW_STREAMER_DOCS_SOFT_LIMIT.c_str());
    }
  }
  if (ailego_unlikely(!shared_mutex_.try_lock_shared())) {
    LOG_ERROR("Cannot add vector while dumping index");
    (*stats_.mutable_discarded_count())++;
    return IndexError_Unsupported;
  }
  AILEGO_DEFER([&]() { shared_mutex_.unlock_shared(); });

  //! The quantized graph region holds each node's neighbor codes, so this
  //! insert invalidates it: the next flush rebuilds the region, and until then
  //! searches fall back to the per-candidate path.
  if (ailego_unlikely(entity_->qg_ready())) {
    ret = invalidate_qg();
    if (ailego_unlikely(ret != 0)) {
      (*stats_.mutable_discarded_count())++;
      return ret;
    }
  }

  ctx->clear();
  ctx->bind_dist_space(add_distance_, add_batch_distance_, provider_);
  ctx->update_dist_caculator_quantizer(add_quantizer_, /*symmetric=*/true);
  ctx->check_need_adjuct_ctx(entity_->doc_cnt());

  //! use the original vector from provider as the build query
  IndexStorage::MemoryBlock original_query_block;
  if (ailego_unlikely(provider_ != nullptr)) {
    ret = provider_->get_vector(pkey, original_query_block);
    if (ailego_unlikely(ret != 0)) {
      LOG_ERROR("Failed to get original vector from provider, key=%llu",
                static_cast<unsigned long long>(pkey));
      (*stats_.mutable_discarded_count())++;
      return ret;
    }
    ctx->reset_query_raw(original_query_block.data(), provider_meta_);
  } else {
    ctx->reset_query(query, meta_);
  }

  if (metric_->support_train()) {
    const std::lock_guard<std::mutex> lk(mutex_);
    ret = metric_->train(query, meta_.dimension());
    if (ailego_unlikely(ret != 0)) {
      LOG_ERROR("Hnsw streamer metric train failed");
      (*stats_.mutable_discarded_count())++;
      return ret;
    }
  }

  // Encode the raw vector into a PQ code for storage and for symmetric
  // (code-vs-code) distance computation during graph construction when
  // the turbo quantizer works internally.
  const void *store_data = query;
  std::string pq_code_buf;
  if (use_internal_quantizer(add_quantizer_, qmeta)) {
    pq_code_buf.resize(add_quantizer_->quantized_datapoint_vector_length());
    add_quantizer_->quantize_data(query, &pq_code_buf[0]);
    store_data = pq_code_buf.data();
  }
  // With a provider the build query is the original provider vector bound
  // above; only the non-provider path rebinds it to the stored data.
  if (provider_ == nullptr) {
    ctx->reset_query(store_data, meta_);
  }

  level_t level = alg_->get_random_level();
  node_id_t id;
  ret = entity_->add_vector(level, pkey, store_data, &id);
  if (ailego_unlikely(ret != 0)) {
    LOG_ERROR("Hnsw streamer add vector failed");
    (*stats_.mutable_discarded_count())++;
    return ret;
  }

  ret = alg_->add_node(id, level, ctx);
  if (ailego_unlikely(ret != 0)) {
    LOG_ERROR("Hnsw steamer add node failed");
    (*stats_.mutable_discarded_count())++;
    return ret;
  }

  if (ailego_unlikely(ctx->error())) {
    (*stats_.mutable_discarded_count())++;
    return IndexError_Runtime;
  }
  (*stats_.mutable_added_count())++;

  return 0;
}


int HnswStreamer::search_impl(const void *query, const IndexQueryMeta &qmeta,
                              IndexStreamer::Context::Pointer &context) const {
  return search_impl(query, qmeta, 1, context);
}

//! Similarity search
int HnswStreamer::search_impl(const void *query, const IndexQueryMeta &qmeta,
                              uint32_t count,
                              IndexStreamer::Context::Pointer &context) const {
  int ret = check_query_params(query, qmeta, search_quantizer_ != nullptr);
  if (ailego_unlikely(ret != 0)) {
    return ret;
  }
  HnswContext *ctx = dynamic_cast<HnswContext *>(context.get());
  ailego_do_if_false(ctx) {
    LOG_ERROR("Cast context to HnswContext failed");
    return IndexError_Cast;
  }

  if (entity_->doc_cnt() <= ctx->get_bruteforce_threshold()) {
    return search_bf_impl(query, qmeta, count, context);
  }

  if (ctx->magic() != magic_) {
    //! context is created by another searcher or streamer
    ret = update_context(ctx);
    if (ret != 0) {
      return ret;
    }
  }

  ctx->clear();
  //! search always uses the vectors stored in the entity
  ctx->bind_dist_space(search_distance_, search_batch_distance_, nullptr);
  ctx->update_dist_caculator_quantizer(search_quantizer_,
                                       /*symmetric=*/false);
  ctx->resize_results(count);
  ctx->check_need_adjuct_ctx(entity_->doc_cnt());
  // Convert raw fp32 queries into the quantized query form (e.g. a PQ LUT)
  // internally when the storage holds PQ codes.
  const bool internal_quant = use_internal_quantizer(search_quantizer_, qmeta);
  std::string quant_query_buf;
  for (size_t q = 0; q < count; ++q) {
    const void *qdata = query;
    if (internal_quant) {
      quant_query_buf.resize(
          search_quantizer_->quantized_query_vector_length());
      search_quantizer_->quantize_query(query, &quant_query_buf[0]);
      qdata = quant_query_buf.data();
    }
    ctx->reset_query(qdata, meta_);
    ret = alg_->search(ctx);
    if (ailego_unlikely(ret != 0)) {
      LOG_ERROR("Hnsw searcher fast search failed");
      return ret;
    }
    ctx->topk_to_result(q);
    query = static_cast<const char *>(query) + qmeta.element_size();
  }

  if (ailego_unlikely(ctx->error())) {
    return IndexError_Runtime;
  }

  return 0;
}

void HnswStreamer::print_debug_info() {
  for (node_id_t id = 0; id < entity_->doc_cnt(); ++id) {
    if (entity_->get_key(id) == kInvalidKey) {
      continue;
    }
    Neighbors neighbours = entity_->get_neighbors(0, id);
    std::cout << "node: " << id << "; ";
    if (neighbours.size() == 0) std::cout << std::endl;
    for (uint32_t i = 0; i < neighbours.size(); ++i) {
      std::cout << neighbours[i];

      if (i == neighbours.size() - 1) {
        std::cout << std::endl;
      } else {
        std::cout << ", ";
      }
    }
  }

  // entity_->print_key_map();
}

int HnswStreamer::search_bf_impl(
    const void *query, const IndexQueryMeta &qmeta,
    IndexStreamer::Context::Pointer &context) const {
  return search_bf_impl(query, qmeta, 1, context);
}

int HnswStreamer::search_bf_impl(
    const void *query, const IndexQueryMeta &qmeta, uint32_t count,
    IndexStreamer::Context::Pointer &context) const {
  int ret = check_query_params(query, qmeta, search_quantizer_ != nullptr);
  if (ailego_unlikely(ret != 0)) {
    return ret;
  }
  HnswContext *ctx = dynamic_cast<HnswContext *>(context.get());
  ailego_do_if_false(ctx) {
    LOG_ERROR("Cast context to HnswContext failed");
    return IndexError_Cast;
  }
  if (ctx->magic() != magic_) {
    //! context is created by another searcher or streamer
    ret = update_context(ctx);
    if (ret != 0) {
      return ret;
    }
  }

  ctx->clear();
  //! search always uses the vectors stored in the entity
  ctx->bind_dist_space(search_distance_, search_batch_distance_, nullptr);
  ctx->update_dist_caculator_quantizer(search_quantizer_,
                                       /*symmetric=*/false);
  ctx->resize_results(count);

  // Convert raw fp32 queries into the quantized query form (e.g. a PQ LUT)
  // internally when the storage holds PQ codes.
  const bool internal_quant = use_internal_quantizer(search_quantizer_, qmeta);
  std::string quant_query_buf;

  if (ctx->group_by_search()) {
    if (!ctx->group_by().is_valid()) {
      LOG_ERROR("Invalid group-by function");
      return IndexError_InvalidArgument;
    }

    std::function<std::string(node_id_t)> group_by = [&](node_id_t id) {
      return ctx->group_by()(entity_->get_key(id));
    };

    for (size_t q = 0; q < count; ++q) {
      const void *qdata = query;
      if (internal_quant) {
        quant_query_buf.resize(
            search_quantizer_->quantized_query_vector_length());
        search_quantizer_->quantize_query(query, &quant_query_buf[0]);
        qdata = quant_query_buf.data();
      }
      ctx->reset_query(qdata, meta_);
      ctx->group_topk_heaps().clear();

      for (node_id_t id = 0; id < entity_->doc_cnt(); ++id) {
        if (entity_->get_key(id) == kInvalidKey) {
          continue;
        }

        if (!ctx->filter().is_valid() || !ctx->filter()(entity_->get_key(id))) {
          dist_t dist = ctx->dist_calculator().batch_dist(id);

          std::string group_id = group_by(id);

          auto &topk_heap = ctx->group_topk_heaps()[group_id];
          if (topk_heap.empty()) {
            topk_heap.limit(ctx->group_topk());
          }
          topk_heap.emplace(id, dist);
        }
      }
      ctx->topk_to_result(q);
      query = static_cast<const char *>(query) + qmeta.element_size();
    }
  } else {
    auto &filter = ctx->filter();
    auto &topk = ctx->topk_heap();

    for (size_t q = 0; q < count; ++q) {
      const void *qdata = query;
      if (internal_quant) {
        quant_query_buf.resize(
            search_quantizer_->quantized_query_vector_length());
        search_quantizer_->quantize_query(query, &quant_query_buf[0]);
        qdata = quant_query_buf.data();
      }
      ctx->reset_query(qdata, meta_);
      topk.clear();
      for (node_id_t id = 0; id < entity_->doc_cnt(); ++id) {
        if (entity_->get_key(id) == kInvalidKey) {
          continue;
        }

        if (!filter.is_valid() || !filter(entity_->get_key(id))) {
          dist_t dist = ctx->dist_calculator().batch_dist(id);
          topk.emplace(id, dist);
        }
      }
      ctx->topk_to_result(q);
      query = static_cast<const char *>(query) + qmeta.element_size();
    }
  }

  if (ailego_unlikely(ctx->error())) {
    return IndexError_Runtime;
  }

  return 0;
}

int HnswStreamer::search_bf_by_p_keys_impl(
    const void *query, const std::vector<std::vector<uint64_t>> &p_keys,
    const IndexQueryMeta &qmeta, uint32_t count,
    Context::Pointer &context) const {
  int ret = check_query_params(query, qmeta, search_quantizer_ != nullptr);
  if (ailego_unlikely(ret != 0)) {
    return ret;
  }

  if (ailego_unlikely(p_keys.size() != count)) {
    LOG_ERROR("The size of p_keys is not equal to count");
    return IndexError_InvalidArgument;
  }

  HnswContext *ctx = dynamic_cast<HnswContext *>(context.get());
  ailego_do_if_false(ctx) {
    LOG_ERROR("Cast context to HnswContext failed");
    return IndexError_Cast;
  }
  if (ctx->magic() != magic_) {
    //! context is created by another searcher or streamer
    ret = update_context(ctx);
    if (ret != 0) {
      return ret;
    }
  }

  ctx->clear();
  //! search always uses the vectors stored in the entity
  ctx->bind_dist_space(search_distance_, search_batch_distance_, nullptr);
  ctx->update_dist_caculator_quantizer(search_quantizer_,
                                       /*symmetric=*/false);
  ctx->resize_results(count);

  // Convert raw fp32 queries into the quantized query form (e.g. a PQ LUT)
  // internally when the storage holds PQ codes.
  const bool internal_quant = use_internal_quantizer(search_quantizer_, qmeta);
  std::string quant_query_buf;

  if (ctx->group_by_search()) {
    if (!ctx->group_by().is_valid()) {
      LOG_ERROR("Invalid group-by function");
      return IndexError_InvalidArgument;
    }

    std::function<std::string(node_id_t)> group_by = [&](node_id_t id) {
      return ctx->group_by()(entity_->get_key(id));
    };

    for (size_t q = 0; q < count; ++q) {
      const void *qdata = query;
      if (internal_quant) {
        quant_query_buf.resize(
            search_quantizer_->quantized_query_vector_length());
        search_quantizer_->quantize_query(query, &quant_query_buf[0]);
        qdata = quant_query_buf.data();
      }
      ctx->reset_query(qdata, meta_);
      ctx->group_topk_heaps().clear();

      for (size_t idx = 0; idx < p_keys[q].size(); ++idx) {
        uint64_t pk = p_keys[q][idx];
        if (!ctx->filter().is_valid() || !ctx->filter()(pk)) {
          node_id_t id = entity_->get_id(pk);
          if (id != kInvalidNodeId) {
            dist_t dist = ctx->dist_calculator().batch_dist(id);
            std::string group_id = group_by(id);

            auto &topk_heap = ctx->group_topk_heaps()[group_id];
            if (topk_heap.empty()) {
              topk_heap.limit(ctx->group_topk());
            }
            topk_heap.emplace(id, dist);
          }
        }
      }
      ctx->topk_to_result(q);
      query = static_cast<const char *>(query) + qmeta.element_size();
    }
  } else {
    auto &filter = ctx->filter();
    auto &topk = ctx->topk_heap();

    for (size_t q = 0; q < count; ++q) {
      const void *qdata = query;
      if (internal_quant) {
        quant_query_buf.resize(
            search_quantizer_->quantized_query_vector_length());
        search_quantizer_->quantize_query(query, &quant_query_buf[0]);
        qdata = quant_query_buf.data();
      }
      ctx->reset_query(qdata, meta_);
      topk.clear();
      for (size_t idx = 0; idx < p_keys[q].size(); ++idx) {
        key_t pk = p_keys[q][idx];
        if (!filter.is_valid() || !filter(pk)) {
          node_id_t id = entity_->get_id(pk);
          if (id != kInvalidNodeId) {
            dist_t dist = ctx->dist_calculator().batch_dist(id);
            topk.emplace(id, dist);
          }
        }
      }
      ctx->topk_to_result(q);
      query = static_cast<const char *>(query) + qmeta.element_size();
    }
  }

  if (ailego_unlikely(ctx->error())) {
    return IndexError_Runtime;
  }

  return 0;
}


INDEX_FACTORY_REGISTER_STREAMER(HnswStreamer);

}  // namespace core
}  // namespace zvec
