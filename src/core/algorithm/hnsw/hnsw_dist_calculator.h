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

#include <turbo/quantizer/quantizer.h>
#include <zvec/core/framework/index_meta.h>
#include "hnsw_entity.h"

namespace zvec {
namespace core {

//! Dist calculator used by HNSW. When a turbo Quantizer is attached,
//! distances are computed directly via the quantizer's calc_* APIs:
//! search (asymmetric) uses calc_distance_dp_query / _batch between a
//! pre-quantized query and stored codes, while graph construction
//! (symmetric) uses calc_distance_dp_dp between stored codes. Without
//! a quantizer it falls back to the IndexMetric distance handles,
//! keeping the legacy behavior unchanged.
class HnswDistCalculator {
 public:
  typedef std::shared_ptr<HnswDistCalculator> Pointer;

 public:
  enum DistType {
    DIST_NONE = 0,
    DIST_DENSE = 1,
    DIST_HYBRID = 2,
    DIST_SPARSE = 3
  };

 public:
  //! Constructor
  HnswDistCalculator(const HnswEntity *entity,
                     const IndexMetric::Pointer &metric, uint32_t dim)
      : entity_(entity),
        distance_(metric->distance()),
        batch_distance_(metric->batch_distance()),
        query_(nullptr),
        dim_(dim),
        compare_cnt_(0) {}

  //! Constructor
  HnswDistCalculator(const HnswEntity *entity,
                     const IndexMetric::Pointer &metric, uint32_t dim,
                     const void *query)
      : entity_(entity),
        distance_(metric->distance()),
        batch_distance_(metric->batch_distance()),
        query_(query),
        dim_(dim),
        compare_cnt_(0) {}

  //! Constructor
  HnswDistCalculator(const HnswEntity *entity,
                     const IndexMetric::Pointer &metric)
      : entity_(entity),
        distance_(metric->distance()),
        batch_distance_(metric->batch_distance()),
        query_(nullptr),
        dim_(0),
        compare_cnt_(0) {}

  //! Constructor with a turbo quantizer and an IndexMetric fallback
  HnswDistCalculator(const HnswEntity *entity,
                     const zvec::turbo::Quantizer::Pointer &quantizer,
                     const IndexMetric::Pointer &metric, uint32_t dim)
      : entity_(entity),
        quantizer_(quantizer),
        distance_(metric->distance()),
        batch_distance_(metric->batch_distance()),
        query_(nullptr),
        dim_(dim),
        compare_cnt_(0) {}

  void update(const HnswEntity *entity, const IndexMetric::Pointer &metric) {
    entity_ = entity;
    distance_ = metric->distance();
    batch_distance_ = metric->batch_distance();
  }

  void update(const HnswEntity *entity, const IndexMetric::Pointer &metric,
              uint32_t dim) {
    entity_ = entity;
    distance_ = metric->distance();
    batch_distance_ = metric->batch_distance();
    dim_ = dim;
  }

  void update(const HnswEntity *entity,
              const zvec::turbo::Quantizer::Pointer &quantizer,
              const IndexMetric::Pointer &metric, uint32_t dim) {
    entity_ = entity;
    quantizer_ = quantizer;
    distance_ = metric->distance();
    batch_distance_ = metric->batch_distance();
    dim_ = dim;
  }

  inline void update_distance(
      const IndexMetric::MatrixDistance &distance,
      const IndexMetric::MatrixBatchDistance &batch_distance) {
    distance_ = distance;
    batch_distance_ = batch_distance;
  }

  //! Replace the turbo quantizer. `symmetric` selects the distance
  //! semantics: true for graph construction (dp-vs-dp SDC, the bound
  //! query is a stored code), false for search (dp-vs-query ADC, the
  //! bound query is a pre-quantized query e.g. a PQ LUT). Pass a null
  //! quantizer to fall back to the IndexMetric path.
  inline void update_quantizer(zvec::turbo::Quantizer::Pointer quantizer,
                               bool symmetric) {
    quantizer_ = std::move(quantizer);
    symmetric_ = symmetric;
  }

  inline bool has_quantizer() const {
    return quantizer_ != nullptr;
  }

  //! Reset query vector data
  inline void reset_query(const void *query) {
    error_ = false;
    query_ = query;
  }

  //! Returns distance between two stored vectors (pairwise), computed
  //! with dp-vs-dp distance (SDC when a PQ quantizer is attached).
  inline dist_t dist(const void *vec_lhs, const void *vec_rhs) {
    if (ailego_unlikely(vec_lhs == nullptr || vec_rhs == nullptr)) {
      LOG_ERROR("Nullptr of dense vector");
      error_ = true;
      return 0.0f;
    }

    if (quantizer_ != nullptr) {
      return quantizer_->calc_distance_dp_dp(vec_lhs, vec_rhs);
    }

    float score{0.0f};

    distance_(vec_lhs, vec_rhs, dim_, &score);

    return score;
  }

  //! Returns distance between the bound query and an already-fetched
  //! vector, without touching compare_cnt_.
  inline dist_t dist_vs_query(const void *vec) {
    if (ailego_unlikely(vec == nullptr || query_ == nullptr)) {
      LOG_ERROR("Nullptr of dense vector or query");
      error_ = true;
      return 0.0f;
    }

    if (quantizer_ != nullptr) {
      return symmetric_ ? quantizer_->calc_distance_dp_dp(query_, vec)
                        : quantizer_->calc_distance_dp_query(vec, query_);
    }

    float score{0.0f};

    distance_(vec, query_, dim_, &score);

    return score;
  }

  //! Returns distance between query and vec.
  inline dist_t dist(const void *vec) {
    compare_cnt_++;

    return dist_vs_query(vec);
  }

  //! Return distance between query and node id.
  inline dist_t dist(node_id_t id) {
    compare_cnt_++;
    IndexStorage::MemoryBlock vec_block;
    int ret = entity_->get_vector(id, vec_block);
    if (ailego_unlikely(ret != 0)) {
      LOG_ERROR("Get nullptr vector, id=%u", id);
      error_ = true;
      return 0.0f;
    }
    const void *feat = vec_block.data();
    if (ailego_unlikely(feat == nullptr)) {
      LOG_ERROR("Get nullptr vector, id=%u", id);
      error_ = true;
      return 0.0f;
    }

    return dist_vs_query(feat);
  }

  //! Return dist node lhs between node rhs
  inline dist_t dist(node_id_t lhs, node_id_t rhs) {
    compare_cnt_++;


    IndexStorage::MemoryBlock vec_block_feat;
    int ret = entity_->get_vector(lhs, vec_block_feat);
    if (ailego_unlikely(ret != 0)) {
      LOG_ERROR("Get nullptr vector, id=%u", lhs);
      error_ = true;
      return 0.0f;
    }
    const void *feat = vec_block_feat.data();

    IndexStorage::MemoryBlock vec_block_query;
    ret = entity_->get_vector(rhs, vec_block_query);
    if (ailego_unlikely(ret != 0)) {
      LOG_ERROR("Get nullptr vector, id=%u", rhs);
      error_ = true;
      return 0.0f;
    }
    const void *query = vec_block_query.data();
    if (ailego_unlikely(feat == nullptr || query == nullptr)) {
      LOG_ERROR("Get nullptr vector");
      error_ = true;
      return 0.0f;
    }

    return dist(feat, query);
  }

  dist_t operator()(const void *vec) {
    return dist(vec);
  }

  dist_t operator()(node_id_t i) {
    return dist(i);
  }

  dist_t operator()(node_id_t lhs, node_id_t rhs) {
    return dist(lhs, rhs);
  }

  void batch_dist(const void **vecs, size_t num, dist_t *distances) {
    compare_cnt_++;

    if (quantizer_ != nullptr) {
      if (!symmetric_) {
        //! Batch ADC between stored codes and the pre-quantized query
        quantizer_->calc_distance_dp_query_batch(
            vecs, static_cast<int>(num), query_, distances);
      } else {
        //! No batch SDC kernel: per-vector dp-vs-dp distance
        for (size_t i = 0; i < num; ++i) {
          distances[i] = quantizer_->calc_distance_dp_dp(query_, vecs[i]);
        }
      }
      return;
    }

    batch_distance_(vecs, query_, num, dim_, distances);
  }

  inline dist_t batch_dist(node_id_t id) {
    compare_cnt_++;

    IndexStorage::MemoryBlock vec_block;
    int ret = entity_->get_vector(id, vec_block);
    if (ailego_unlikely(ret != 0)) {
      LOG_ERROR("Get nullptr vector, id=%u", id);
      error_ = true;
      return 0.0f;
    }
    const void *feat = vec_block.data();
    if (ailego_unlikely(feat == nullptr)) {
      LOG_ERROR("Get nullptr vector, id=%u", id);
      error_ = true;
      return 0.0f;
    }

    if (quantizer_ != nullptr) {
      return dist_vs_query(feat);
    }

    dist_t score = 0;
    batch_distance_(&feat, query_, 1, dim_, &score);

    return score;
  }

  inline void clear() {
    compare_cnt_ = 0;
    error_ = false;
  }

  inline void clear_compare_cnt() {
    compare_cnt_ = 0;
  }

  inline bool error() const {
    return error_;
  }

  //! Get distances compute times
  inline uint32_t compare_cnt() const {
    return compare_cnt_;
  }

  inline uint32_t dimension() const {
    return dim_;
  }

 private:
  HnswDistCalculator(const HnswDistCalculator &) = delete;
  HnswDistCalculator &operator=(const HnswDistCalculator &) = delete;

 private:
  const HnswEntity *entity_;

  //! Optional turbo quantizer; when set, distances go through its
  //! calc_* APIs instead of the IndexMetric handles below.
  zvec::turbo::Quantizer::Pointer quantizer_{};
  //! Distance semantics of the attached quantizer: true = graph
  //! construction (dp-vs-dp), false = search (dp-vs-query).
  bool symmetric_{false};

  IndexMetric::MatrixDistance distance_;
  IndexMetric::MatrixBatchDistance batch_distance_;

  const void *query_;
  uint32_t dim_;

  uint32_t compare_cnt_;  // record distance compute times
  // uint32_t compare_cnt_batch_;  // record batch distance compute time
  bool error_{false};
};

}  // namespace core
}  // namespace zvec
