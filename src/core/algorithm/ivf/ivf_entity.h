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

#include <core/quantizer/quantizer_params.h>
#include <turbo/quantizer/common/packed_code_quantizer.h>
#include <turbo/quantizer/quantizer.h>
#include <zvec/core/framework/index_framework.h>
#include "metric/metric_params.h"
#include "ivf_distance_calculator.h"
#include "ivf_index_format.h"
#include "ivf_params.h"

namespace zvec {
namespace core {

/*! IVF Entity
 */
class IVFEntity {
 public:
  typedef std::shared_ptr<IVFEntity> Pointer;

  class IVFReformerWrapper;

  //! Constructor
  IVFEntity() {}

  //! Destructor
  virtual ~IVFEntity() {}

  //! Disable them
  IVFEntity(const IVFEntity &) = delete;
  IVFEntity &operator=(const IVFEntity &) = delete;

  //! load the index from container
  virtual int load(const IndexStorage::Pointer &container);

  //! search in inverted list with filter
  int search(size_t inverted_list_id, const void *query,
             const IndexFilter &filter, uint32_t *scan_count,
             IndexDocumentHeap *heap, IndexContext::Stats *context_stats) const;

  //! search in inverted list without filter
  int search(size_t inverted_list_id, const void *query, uint32_t *scan_count,
             IndexDocumentHeap *heap, IndexContext::Stats *context_stats) const;

  //! search all inverted list with filter
  int search(const void *query, const IndexFilter &filter,
             IndexDocumentHeap *heap, IndexContext::Stats *context_stats) const;

  //! search all inverted list without filter
  int search(const void *query, IndexDocumentHeap *heap,
             IndexContext::Stats *context_stats) const;

  //! Clone the entity
  virtual IVFEntity::Pointer clone(void) const;

  //! Clone the entity
  IVFEntity::Pointer clone(const IVFEntity::Pointer &entity) const;

  //! Retrieve the primary keys by local id in heap
  int retrieve_keys(IndexDocumentHeap *heap) const {
    for (auto &it : (*heap)) {
      uint64_t key = this->get_key(it.index());
      if (key == kInvalidKey) {
        return IndexError_ReadData;
      }
      it.set_key(key);
    }

    return 0;
  }

  //! Retrieve the total vectors in the index
  size_t vector_count(void) const {
    return header_.total_vector_count;
  }

  //! Retrieve the inverted list count
  size_t inverted_list_count(void) const {
    return header_.inverted_list_count;
  }

  //! Retrieve block size of the inverted vector
  size_t inverted_block_size(void) const {
    return header_.block_size;
  }

  //! Retrieve the vectors count in one block
  size_t block_vector_count(void) const {
    return header_.block_vector_count;
  }

  //! Retrieve IndexMeta of the inverted index
  const IndexMeta &meta(void) const {
    return meta_;
  }

  //! Retrieve a block of vectors
  const void *read_block(size_t inverted_list_id, size_t local_block_id,
                         size_t *vecs_count) const {
    auto iv_meta = this->inverted_list_meta(inverted_list_id);
    if (!iv_meta || local_block_id >= iv_meta->block_count) {
      LOG_ERROR("Failed to read inverted list, listId=%zu blockIdx=%zu",
                inverted_list_id, local_block_id);
      return nullptr;
    }

    size_t block_vecs = header_.block_vector_count;
    *vecs_count = std::min(block_vecs,
                           iv_meta->vector_count - local_block_id * block_vecs);
    ailego_assert_with(*vecs_count <= header_.block_vector_count,
                       "invalid vecs");
    const size_t off = iv_meta->offset + local_block_id * header_.block_size;
    const size_t size = *vecs_count * meta_.element_size();
    const void *data = nullptr;
    if (inverted_->read(off, &data, size) != size) {
      LOG_ERROR("Failed to read block off=%zu size=%zu", off, size);
      return nullptr;
    }

    return data;
  }

  //! Retrieve the inverted list meta
  const InvertedListMeta *inverted_list_meta(size_t inverted_list_id) const {
    const void *data = nullptr;
    const size_t size = sizeof(InvertedListMeta);
    const size_t offset = inverted_list_id * size;
    if (inverted_meta_->read(offset, &data, size) != size) {
      LOG_ERROR("Failed to read inverted meta, id=%zu, size=%zu",
                inverted_list_id, size);
      return nullptr;
    }

    return static_cast<const InvertedListMeta *>(data);
  }

  //! Retrieve the keys by consecutive local ids
  const uint64_t *get_keys(size_t id, size_t count) const {
    const void *data = nullptr;
    const size_t offset = id * sizeof(uint64_t);
    const size_t size = count * sizeof(uint64_t);
    if (keys_->read(offset, &data, size) != size) {
      LOG_ERROR("Failed to read keys, id=%zu, size=%zu", id, size);
      return nullptr;
    }

    return static_cast<const uint64_t *>(data);
  }

  //! Retrieve the key by local id
  uint64_t get_key(size_t id) const {
    const void *data = nullptr;
    const size_t offset = id * sizeof(uint64_t);
    const size_t size = sizeof(uint64_t);
    if (keys_->read(offset, &data, size) != size) {
      LOG_ERROR("Failed to read key, id=%zu", id);
      return kInvalidKey;
    }

    return *static_cast<const uint64_t *>(data);
  }

  //! Retrieve the key-order mapping (sorted rank -> local_id).
  //! mapping[rank] is the local_id of the vector with the rank-th smallest
  //! key. Returns nullptr if mapping segment is unavailable.
  const uint32_t *get_key_order_mapping() const {
    if (!mapping_) return nullptr;
    const void *data = nullptr;
    const size_t size = vector_count() * sizeof(uint32_t);
    if (mapping_->read(0, &data, size) != size) {
      return nullptr;
    }
    return static_cast<const uint32_t *>(data);
  }

  //! Retrieve vector by local id
  const void *get_vector(size_t id) const;

  //! Retrieve vector by local id
  const void *get_vector_by_key(uint64_t key) const;

  int get_vector(size_t id, IndexStorage::MemoryBlock &block) const;

  int get_vector_by_key(uint64_t key, IndexStorage::MemoryBlock &block) const;

  uint32_t key_to_id(uint64_t key) const;

  //! Transform a query
  int transform(const void *query, const IndexQueryMeta &qmeta,
                const void **out, IndexQueryMeta *ometa) const {
    if (quantizer_ && !use_residual_) {
      return this->transform_quantized(query, qmeta, 1, out, ometa);
    }
    //! Residual mode: a global LUT is invalid (each list needs its own
    //! residual query), pass the query through and let search() build the
    //! per-list LUT.
    return reformer_.transform(query, qmeta, out, ometa);
  }

  //! Transform queries
  int transform(const void *query, const IndexQueryMeta &qmeta, uint32_t count,
                const void **out, IndexQueryMeta *ometa) const {
    if (quantizer_ && !use_residual_) {
      return this->transform_quantized(query, qmeta, count, out, ometa);
    }
    return reformer_.transform(query, qmeta, count, out, ometa);
  }

  //! Normalize the score in query part
  void normalize(size_t qidx, IndexDocumentHeap *heap) const {
    return reformer_.normalize(qidx, heap);
  }

  //! Retrieve the value for each inverted list to multiply for normalizing
  float inverted_list_normalize_value(size_t inverted_list_id) const {
    if (norm_value_ != 0.0f) {
      return norm_value_;
    }

    // ailego_assert_with(integer_quantizer_params_, "nullptr");
    if (integer_quantizer_params_ != nullptr) {
      const void *data = nullptr;
      size_t size = sizeof(InvertedIntegerQuantizerParams);
      size_t off = inverted_list_id * size;
      if (integer_quantizer_params_->read(off, &data, size) != size) {
        LOG_ERROR("Failed to read data from segment, off=%zu", off);
        return 1.0f;
      }
      auto scale =
          static_cast<const InvertedIntegerQuantizerParams *>(data)->scale;
      return this->convert_to_normalize_value(scale);
    }

    return 1.0f;
  }

  //! Check whether the feature segment exist
  bool has_orignal_feature() const {
    return !!features_;
  }

  //! Retrieve reformer
  const IVFReformerWrapper &reformer(void) const {
    return reformer_;
  }

  //! Attach a turbo quantizer. When set, inverted codes are decoded with
  //! the quantizer instead of the metric-based distance calculator. The
  //! entity only downcasts it to discover the optional precomputed-table
  //! capability (see precompute_quantizer_of() in ivf_entity.cc) and the
  //! packed-code capability (cached here once, so the search hot path
  //! needs no cast/branch on quantizer type).
  //! @param index_meta original vector-space meta: quantizers may rewrite
  //!        their own meta to the code representation, so the residual
  //!        query meta must be derived from the index meta instead.
  void set_quantizer(zvec::turbo::Quantizer::Pointer quantizer,
                     const IndexMeta &index_meta) {
    quantizer_ = std::move(quantizer);
    packed_quantizer_ = dynamic_cast<const zvec::turbo::PackedCodeQuantizer *>(
        quantizer_.get());
    if (quantizer_ && use_residual_) {
      //! meta_ is a pseudo code meta; derive the residual query meta from
      //! the original vector-space meta (data type/dimension as trained).
      residual_query_meta_.set_meta(index_meta.data_type(),
                                    index_meta.dimension());
      residual_input_element_size_ = index_meta.element_size();
    }
  }

  //! Retrieve the turbo quantizer
  const zvec::turbo::Quantizer::Pointer &quantizer(void) const {
    return quantizer_;
  }

  //! Whether the inverted codes store residuals (v - centroid[label])
  bool use_residual(void) const {
    return use_residual_;
  }

  //! Enable/disable the precomputed residual distance table. Called by
  //! searcher/streamer after load; builds the centroid table through the
  //! turbo quantizer interface. Never fails the load: quantizers that do
  //! not support the table keep the per-list residual path.
  int set_precompute_enabled(bool enable);

  //! Per-query hook for the precomputed residual path: normalize the query
  //! once (Cosine) and cache the query-side distance table. No-op when the
  //! precomputed path is inactive.
  int prepare_query(const void *query) const;

  /*! Index Reformer Wrapper
   *  To transform query in inverted index searching, and normalize the score
   */
  class IVFReformerWrapper {
   public:
    //! Constructor
    IVFReformerWrapper() {}

    //! Assignment
    IVFReformerWrapper &operator=(const IVFReformerWrapper &wrapper) {
      reformer_ = wrapper.reformer_;
      type_ = wrapper.type_;
      buffer_.clear();
      buffer_.shrink_to_fit();
      reciprocal_ = wrapper.reciprocal_;
      return *this;
    }

    //! Initialize
    int init(const IndexMeta &imeta);

    //! Load reformer state (e.g. rotation matrix) from storage
    int load(const IndexStorage::Pointer &storage);

    //! Update
    int update(const IndexMeta &meta);

    //! Transform a query
    int transform(const void *query, const IndexQueryMeta &qmeta,
                  const void **out, IndexQueryMeta *ometa);

    //! Transform queries
    int transform(const void *query, const IndexQueryMeta &qmeta,
                  uint32_t count, const void **out, IndexQueryMeta *ometa);

    //! Convert a record
    virtual int convert(const void *record, const IndexQueryMeta &rmeta,
                        const void **out, IndexQueryMeta *ometa);

    //! Convert records
    virtual int convert(const void *records, const IndexQueryMeta &rmeta,
                        uint32_t count, const void **out,
                        IndexQueryMeta *ometa);

    //! Transform queries
    int transform_gpu(const void *query, const IndexQueryMeta &qmeta,
                      uint32_t count, const void **out, IndexQueryMeta *ometa);

    //! Normalize the score in query part
    void normalize(size_t qidx, IndexDocumentHeap *heap) const;

    //! Normalize the score in query part
    void normalize(size_t qidx, const void *query, const IndexQueryMeta &qmeta,
                   IndexDocumentHeap *heap) const;

   private:
    //! Transform query from fp32 to int8
    void transform(size_t qidx, const float *in, size_t dim, int8_t *out);

    //! Transform query from fp32 to int4
    void transform(size_t qidx, const float *in, size_t dim, uint8_t *out);

   private:
    //! Constants
    enum Type {
      kReformerTpNone = 0,
      kReformerTpInnerProductInt8 = 1,
      kReformerTpInnerProductInt4 = 2,
      kReformerTpInt8 = 3,
      kReformerTpInt4 = 4,
      kReformerTpDefault = 7,
    };

    //! Members
    Type type_{kReformerTpNone};
    IndexReformer::Pointer reformer_{};
    std::string buffer_{};
    float reciprocal_{0.0};        // for int8
    std::vector<float> scales_{};  // for int8 IP
  };

 private:
  //! Load the segment by seg_id in expect_size segment size
  IndexStorage::Segment::Pointer load_segment(const std::string &seg_id,
                                              size_t expect_size) const;

  //! Load the header segment
  int load_header(const IndexStorage::Pointer &container);

  //! Transform queries into quantizer LUTs via the turbo quantizer
  int transform_quantized(const void *query, const IndexQueryMeta &qmeta,
                          uint32_t count, const void **out,
                          IndexQueryMeta *ometa) const;

  //! Residual mode: build the per-list residual query (normalize for Cosine,
  //! then subtract the list centroid) and quantize it into the LUT buffer
  //! residual_query_.
  int prepare_residual_query(size_t inverted_list_id, const void *query) const;

  //! L2-normalize a residual-space vector buffer in-place (fp16 goes
  //! through a fp32 scratch). Shared by the per-list and the precomputed
  //! residual paths.
  void normalize_residual_vec(char *vec) const;

  //! term1 of the residual decomposition: ||q' - c_i||^2 between the
  //! prepared query cached by prepare_query() and the list centroid.
  float compute_centroid_distance(size_t inverted_list_id) const;

  //! Compute distances of a block of codes against a quantized query (LUT)
  void quantized_block_distance(const void *query, const void *block_data,
                                size_t vecs_count, float *distances) const;

  //! Convert the int8 quantizer scale to normalize value
  float convert_to_normalize_value(float scale) const {
    auto v = scale == 0.0 ? 1.0 : (1.0 / scale);
    return !norm_value_sqrt_ ? v : std::sqrt(v);
  }

 protected:
  //! Constants
  static constexpr size_t kBatchBlocks = 10u;

  //! Members
  IndexMeta meta_{};
  mutable IVFReformerWrapper reformer_{};
  zvec::turbo::Quantizer::Pointer quantizer_{};
  //! Cached packed-code capability of quantizer_ (null for gather-style
  //! quantizers); owned by quantizer_, refreshed in set_quantizer().
  const zvec::turbo::PackedCodeQuantizer *packed_quantizer_{nullptr};
  mutable std::string quantized_query_{};  // LUT buffer for turbo quantizer

  //! Residual mode state
  bool use_residual_{false};              //! inverted codes store residuals
  bool residual_normalize_query_{false};  //! original metric is Cosine
  IndexStorage::Segment::Pointer residual_centroids_{};
  mutable std::string residual_query_{};      //! per-list LUT buffer
  mutable std::string residual_query_vec_{};  //! residual query vector buffer
  IndexQueryMeta residual_query_meta_{};      //! meta of the residual space
  size_t residual_input_element_size_{0};     //! centroid/query element size

  //! Precomputed residual distance table state (term2/term3 decomposition);
  //! the table is shared across cloned entities, per-query buffers are not.
  bool precompute_enabled_{false};
  mutable bool precompute_active_{false};
  std::shared_ptr<const std::string> precompute_table_{};
  mutable std::string precompute_query_table_{};   //! term3 LUT, per query
  mutable std::string precompute_merged_query_{};  //! merged per-list LUT
  mutable std::string precompute_query_vec_{};     //! normalized query cache
  mutable const void *precompute_prepared_query_{nullptr};  //! guard pointer
  mutable const void *precompute_prepared_vec_{nullptr};    //! term1 operand
  IVFDistanceCalculator::Pointer calculator_{};
  IndexStorage::Pointer container_{};
  IndexStorage::Segment::Pointer inverted_{};
  IndexStorage::Segment::Pointer inverted_meta_{};
  IndexStorage::Segment::Pointer keys_{};
  IndexStorage::Segment::Pointer offsets_{};
  IndexStorage::Segment::Pointer mapping_{};
  IndexStorage::Segment::Pointer features_{};
  IndexStorage::Segment::Pointer integer_quantizer_params_{};
  mutable std::string vector_{};  // temporary buffer for colomn major order
  float norm_value_{0.0f};  // normalize the inverted vector to orignal score
  bool norm_value_sqrt_{false};  // does the norm value need to sqrt
  InvertedIndexHeader header_;
};

}  // namespace core
}  // namespace zvec
