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
#include <zvec/core/framework/index_metric.h>
#include "hnsw_entity.h"

namespace zvec {
namespace core {

//! Dist calculator used by HNSW. Prefers the turbo Quantizer's
//! DistanceImpl when it is available for the current metric/dtype;
//! otherwise falls back to IndexMetric's distance / batch_distance
//! handles. This keeps HNSW functional for metric/dtype combos that
//! turbo does not yet implement.
class HnswDistCalculator {
 public:
  typedef std::shared_ptr<HnswDistCalculator> Pointer;

 public:
  //! Default constructor (for lazy init via update()).
  HnswDistCalculator()
      : entity_(nullptr), query_(nullptr), dim_(0), compare_cnt_(0) {}

  enum DistType {
    DIST_NONE = 0,
    DIST_DENSE = 1,
    DIST_HYBRID = 2,
    DIST_SPARSE = 3
  };

 public:
  //! Constructor with a turbo quantizer and an IndexMetric fallback.
  //! `dim` is the dimension of the stored vectors. `qmeta_data_type`
  //! is the data type of the raw query accepted by `reset_query`.
  HnswDistCalculator(const HnswEntity *entity,
                     zvec::turbo::Quantizer::Pointer quantizer,
                     IndexMetric::Pointer metric, uint32_t dim,
                     IndexMeta::DataType qmeta_data_type)
      : entity_(entity),
        quantizer_(std::move(quantizer)),
        metric_(std::move(metric)),
        query_(nullptr),
        dim_(dim),
        compare_cnt_(0) {
    qmeta_.set_meta(qmeta_data_type, dim);
    if (metric_) {
      distance_ = metric_->distance();
      batch_distance_ = metric_->batch_distance();
    }
  }

  //! Constructor without dimension (for lazy init via update()).
  HnswDistCalculator(const HnswEntity *entity,
                     zvec::turbo::Quantizer::Pointer quantizer,
                     IndexMetric::Pointer metric)
      : entity_(entity),
        quantizer_(std::move(quantizer)),
        metric_(std::move(metric)),
        query_(nullptr),
        dim_(0),
        compare_cnt_(0) {
    if (metric_) {
      distance_ = metric_->distance();
      batch_distance_ = metric_->batch_distance();
    }
  }

  //! Legacy constructor (no quantizer, metric-only fallback).
  HnswDistCalculator(const HnswEntity *entity,
                     const IndexMetric::Pointer &metric, uint32_t dim)
      : entity_(entity),
        metric_(metric),
        query_(nullptr),
        dim_(dim),
        compare_cnt_(0) {
    if (metric_) {
      distance_ = metric_->distance();
      batch_distance_ = metric_->batch_distance();
    }
  }

  //! Legacy constructor (no quantizer, metric-only fallback).
  HnswDistCalculator(const HnswEntity *entity,
                     const IndexMetric::Pointer &metric)
      : entity_(entity),
        metric_(metric),
        query_(nullptr),
        dim_(0),
        compare_cnt_(0) {
    if (metric_) {
      distance_ = metric_->distance();
      batch_distance_ = metric_->batch_distance();
    }
  }

  void update(const HnswEntity *entity,
              zvec::turbo::Quantizer::Pointer quantizer,
              IndexMetric::Pointer metric) {
    entity_ = entity;
    quantizer_ = std::move(quantizer);
    metric_ = std::move(metric);
    dist_impl_ = zvec::turbo::DistanceImpl{};
    if (metric_) {
      distance_ = metric_->distance();
      batch_distance_ = metric_->batch_distance();
    } else {
      distance_ = nullptr;
      batch_distance_ = nullptr;
    }
  }

  void update(const HnswEntity *entity,
              zvec::turbo::Quantizer::Pointer quantizer,
              IndexMetric::Pointer metric, uint32_t dim,
              IndexMeta::DataType qmeta_data_type) {
    entity_ = entity;
    quantizer_ = std::move(quantizer);
    metric_ = std::move(metric);
    dim_ = dim;
    qmeta_.set_meta(qmeta_data_type, dim);
    dist_impl_ = zvec::turbo::DistanceImpl{};
    if (metric_) {
      distance_ = metric_->distance();
      batch_distance_ = metric_->batch_distance();
    } else {
      distance_ = nullptr;
      batch_distance_ = nullptr;
    }
  }

  //! Legacy update (no quantizer).
  void update(const HnswEntity *entity, const IndexMetric::Pointer &metric) {
    entity_ = entity;
    metric_ = metric;
    dist_impl_ = zvec::turbo::DistanceImpl{};
    if (metric_) {
      distance_ = metric_->distance();
      batch_distance_ = metric_->batch_distance();
    } else {
      distance_ = nullptr;
      batch_distance_ = nullptr;
    }
  }

  //! Legacy update (no quantizer).
  void update(const HnswEntity *entity, const IndexMetric::Pointer &metric,
              uint32_t dim) {
    entity_ = entity;
    metric_ = metric;
    dim_ = dim;
    dist_impl_ = zvec::turbo::DistanceImpl{};
    if (metric_) {
      distance_ = metric_->distance();
      batch_distance_ = metric_->batch_distance();
    } else {
      distance_ = nullptr;
      batch_distance_ = nullptr;
    }
  }

  //! Replace the quantizer used by this calculator. Invalidates the
  //! cached DistanceImpl; caller should follow up with reset_query.
  inline void update_quantizer(zvec::turbo::Quantizer::Pointer quantizer) {
    quantizer_ = std::move(quantizer);
    dist_impl_ = zvec::turbo::DistanceImpl{};
  }

  //! Replace the IndexMetric fallback.
  inline void update_metric(IndexMetric::Pointer metric) {
    metric_ = std::move(metric);
    if (metric_) {
      distance_ = metric_->distance();
      batch_distance_ = metric_->batch_distance();
    } else {
      distance_ = nullptr;
      batch_distance_ = nullptr;
    }
  }

  //! Legacy: update raw distance function pointers.
  inline void update_distance(
      const IndexMetric::MatrixDistance &distance,
      const IndexMetric::MatrixBatchDistance &batch_distance) {
    distance_ = distance;
    batch_distance_ = batch_distance;
  }

  //! Reset query vector data. First quantizes the raw query via the
  //! turbo quantizer, then builds a DistanceImpl from the quantized
  //! form. Falls back to IndexMetric's raw query when turbo does not
  //! support this metric/dtype combination.
  inline void reset_query(const void *query) {
    error_ = false;
    query_ = query;
    if (quantizer_) {
      std::string quantized;
      zvec::core::IndexQueryMeta ometa;
      if (quantizer_->quantize(query, qmeta_, &quantized, &ometa) == 0) {
        dist_impl_ = quantizer_->distance(quantized.data(), ometa);
      } else {
        dist_impl_ = zvec::turbo::DistanceImpl{};
      }
    } else {
      dist_impl_ = zvec::turbo::DistanceImpl{};
    }
  }

  //! Returns distance between two already-quantized vectors (pairwise).
  //! Uses dist_impl_.sym_func() for PQ SDC; falls back to metric distance_.
  inline dist_t dist(const void *vec_lhs, const void *vec_rhs) {
    if (ailego_unlikely(vec_lhs == nullptr || vec_rhs == nullptr)) {
      LOG_ERROR("Nullptr of dense vector");
      error_ = true;
      return 0.0f;
    }

    float score = 0.0f;
    const auto &sym = dist_impl_.sym_func();
    if (sym) {
      sym(vec_lhs, vec_rhs, dist_impl_.dim(), &score);
      return score;
    }
    if (ailego_unlikely(!distance_)) {
      LOG_ERROR("No distance handle available");
      error_ = true;
      return 0.0f;
    }
    distance_(vec_lhs, vec_rhs, dim_, &score);
    return score;
  }

  //! Returns distance between query and vec.
  inline dist_t dist(const void *vec) {
    compare_cnt_++;
    if (ailego_unlikely(vec == nullptr)) {
      LOG_ERROR("Nullptr of dense vector");
      error_ = true;
      return 0.0f;
    }
    if (dist_impl_.valid()) {
      return dist_impl_(vec);
    }
    if (ailego_unlikely(!distance_ || query_ == nullptr)) {
      LOG_ERROR("No distance handle or query available");
      error_ = true;
      return 0.0f;
    }
    float score = 0.0f;
    distance_(vec, query_, dim_, &score);
    return score;
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
    if (dist_impl_.valid()) {
      return dist_impl_(feat);
    }
    if (ailego_unlikely(!distance_ || query_ == nullptr)) {
      LOG_ERROR("No distance handle or query available");
      error_ = true;
      return 0.0f;
    }
    float score = 0.0f;
    distance_(feat, query_, dim_, &score);
    return score;
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
    if (dist_impl_.batch_valid()) {
      dist_impl_.batch(vecs, num, distances);
      return;
    }
    if (batch_distance_ && query_ != nullptr) {
      batch_distance_(vecs, query_, num, dim_, distances);
      return;
    }
    for (size_t i = 0; i < num; ++i) {
      distances[i] = dist(vecs[i]);
    }
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
    if (dist_impl_.batch_valid()) {
      dist_t score = 0;
      const void *feats[1] = {feat};
      dist_impl_.batch(feats, 1, &score);
      return score;
    }
    if (batch_distance_ && query_ != nullptr) {
      dist_t score = 0;
      const void *feats[1] = {feat};
      batch_distance_(feats, query_, 1, dim_, &score);
      return score;
    }
    return dist(feat);
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

  //! Expose the underlying turbo quantizer.
  inline const zvec::turbo::Quantizer::Pointer &quantizer() const {
    return quantizer_;
  }

 private:
  HnswDistCalculator(const HnswDistCalculator &) = delete;
  HnswDistCalculator &operator=(const HnswDistCalculator &) = delete;

 private:
  const HnswEntity *entity_;

  zvec::turbo::Quantizer::Pointer quantizer_{};
  IndexMetric::Pointer metric_{};
  zvec::turbo::DistanceImpl dist_impl_{};
  IndexQueryMeta qmeta_{};

  IndexMetric::MatrixDistance distance_{nullptr};
  IndexMetric::MatrixBatchDistance batch_distance_{nullptr};

  const void *query_;
  uint32_t dim_;

  uint32_t compare_cnt_;
  bool error_{false};
};

}  // namespace core
}  // namespace zvec
