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

#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <turbo/quantizer/quantizer.h>

namespace zvec {
namespace core {

/*! IVF coarse-residual encode/decode facade (faiss-style shared codebook).
 *
 *  IVFResidualCodec owns the whole quantization semantic of the IVF+PQ
 *  coarse+fine scheme: the metric policy (Cosine normalization,
 *  InnerProduct residual+dis0), the residual computation against the coarse
 *  centroid table, the per-list ADC LUT construction, and the codebook
 *  serialization. The IVF index layer holds this facade as an opaque
 *  component and only orchestrates which list/centroid a vector belongs to,
 *  never touching quantizer or metric details directly.
 */
class IVFResidualCodec {
 public:
  typedef std::shared_ptr<IVFResidualCodec> Pointer;

  //! Generic init() param keys (the index layer maps its own config surface
  //! onto these keys and passes everything else through untouched):
  //!   "quantizer_class"  string, factory name, default kDefaultQuantizerClass
  //!   "num_chunk"        int,    PQ chunk count, default 8 (0 means default)
  //!   "use_zero_mean"    bool,   forwarded to the quantizer ONLY when
  //!                              present; omit it on load so deserialize()
  //!                              restores the authoritative value from the
  //!                              codebook blob
  //!   "thread_count"     int,    forwarded to the quantizer when present
  //!   "dim"              int,    overrides meta.dimension() (load side may
  //!                              derive it from the persisted centroids)
  static constexpr const char *kDefaultQuantizerClass = "PqInt8Quantizer";

  /*! Per-query scratch, reused across every probed list of one query.
   *
   *  Buffers keep their capacity, so a context that outlives many queries
   *  allocates only on the first one.  Lifetime rule: call prepare_query()
   *  once per query, then build_list_query() once per probed list.
   */
  struct QueryState {
    //! ADC LUT handed to batch_distance(); num_chunk * ksub floats.
    std::vector<float> lut;
    //! Per-block ADC output; block_vector_count floats, sized by the caller.
    std::vector<float> distances;
    //! Raw query of the current query, borrowed (not owned).
    const float *query{nullptr};
    //! dot(w_m, centroid_{m,j}) for the current query; L2 fast path only.
    std::vector<float> ip_table;
    //! Preprocessed query w (normalize + zero-mean); L2 fast path only.
    std::vector<float> query_buf;
    //! True when `lut` depends on the query only, so build_list_query() must
    //! not touch it (InnerProduct: the LUT is identical for every list).
    bool lut_is_query_only{false};
    //! True when the L2 precomputed-table decomposition is in use for this
    //! query, so build_list_query() only does one madd plus dis0.
    bool use_precomputed{false};
  };


  IVFResidualCodec() {}
  ~IVFResidualCodec() {}

  //! Whether the codec supports the metric (L2/Cosine/InnerProduct only).
  //! The index layer consults this at parse time to fall back gracefully.
  static bool SupportsMetric(const std::string &metric_name);

  //! Derive the metric policy, validate dim/num_chunk, then create and init
  //! the underlying quantizer (encoding is always L2; the metric only
  //! selects the search LUT: InnerProduct for IP, else L2).
  int init(const IndexMeta &meta, const ailego::Params &params);

  //! Take over the fp32 coarse centroid table (nlist * dim). Cosine
  //! centroids are L2-normalized in place (idempotent on load).
  int set_centroids(std::vector<float> centroids, size_t nlist);

  //! Access one coarse centroid.
  const float *centroid(size_t list_id) const {
    return centroids_.data() + list_id * dim_;
  }

  //! Access the whole centroid table (for persistence).
  const std::vector<float> &centroids() const {
    return centroids_;
  }

  //! Compute residual = (unit(v) if Cosine else v) - centroid into out (dim).
  void compute_residual(const float *vec, const float *centroid,
                        float *out) const;

  //! Train the shared codebook on a holder of fp32 residuals.
  int train(IndexHolder::Pointer holder) {
    return quantizer_->train(std::move(holder));
  }

  //! Encode one datapoint of list `list_id`: residual + quantize in one step.
  void encode(const float *vec, size_t list_id, uint8_t *code) const;

  //! Build the query-only part of the ADC state once per query; this must
  //! precede every build_list_query() call for the same query.
  //! - IP (faiss residual+dis0): the LUT is built here from the RAW query and
  //!   is identical for every list, so build_list_query() only supplies dis0.
  //! - L2/Cosine: the LUT depends on the per-list residual, so only the query
  //!   pointer and the buffers are set up here.
  int prepare_query(const void *query, QueryState *state) const;

  //! Build the per-list ADC query state: leave `state->lut` ready for
  //! batch_distance() and set `dis0` to the per-list constant added to every
  //! ADC distance.
  //! - L2/Cosine: LUT on the residual r = (unit(q) if Cosine else q) - c,
  //!   dis0 = 0 (residual ADC directly approximates the metric distance).
  //! - IP: LUT already built by prepare_query(); the coarse center contributes
  //!   dis0 = -<q,c>, so dis0 + ADC = -<q,v>.
  void build_list_query(size_t list_id, QueryState *state, float *dis0) const;

  //! Batched ADC distances over contiguous codes against a prebuilt LUT.
  void batch_distance(const void *codes, int num, size_t stride,
                      const void *lut, float *out) const {
    quantizer_->calc_distance_dp_query_batch_contiguous(codes, num, stride,
                                                        lut, out);
  }

  //! Byte length of one quantized datapoint code.
  size_t code_size() const {
    return quantizer_->quantized_datapoint_vector_length();
  }

  //! Byte length of one quantized query LUT.
  size_t lut_size() const {
    return quantizer_->quantized_query_vector_length();
  }

  uint32_t dim() const {
    return dim_;
  }

  uint32_t num_chunk() const {
    return num_chunk_;
  }

  bool use_zero_mean() const {
    return use_zero_mean_;
  }

  const std::string &quantizer_class() const {
    return quantizer_class_;
  }

  //! Output meta of the underlying quantizer (authoritative code layout:
  //! data type, code dimension, extra meta size). Callers retrieve the
  //! quantized storage meta from here instead of assembling it by hand.
  const IndexMeta &meta() const {
    return quantizer_->meta();
  }

  //! Whether codes must be stored in packed 32-vector blocks (FastScan).
  bool requires_packed_codes() const {
    return quantizer_->requires_packed_codes();
  }

  //! Underlying quantizer (for storage-layer block packing only).
  const turbo::Quantizer::Pointer &quantizer() const {
    return quantizer_;
  }

  //! Serialize the codebook into a self-describing blob.
  int serialize_codebook(std::string *blob) const {
    return quantizer_->serialize(blob);
  }

  //! Restore the codebook (and its authoritative config) from a blob.
  int deserialize_codebook(const void *data, size_t len) {
    return quantizer_->deserialize(data, len);
  }

  /*! Precompute the list-dependent half of the L2 ADC table (faiss
   *  use_precomputed_table=1).
   *
   *  With r' = (v - c_i) - mu the encoded residual and w = x - mu the
   *  preprocessed query,
   *
   *    ||x - v_hat||^2 = ||w - c_i||^2
   *                      + sum_m [ ||s_mj||^2 + 2<c_im, s_mj> - 2<w_m, s_mj> ]
   *
   *  so table_[i][m][j] = ||s_mj||^2 + 2<c_im, s_mj> depends on the list and
   *  the codebook only, and the per-list LUT collapses to a single
   *  lut = table_[i] - 2 * ip_table madd.
   *
   *  Only valid for plain L2: Cosine re-normalizes the residual inside
   *  quantize_query(), which is non-linear and breaks the expansion, and IP
   *  does not need it (its LUT is already query-only). Must be called after
   *  set_centroids() and deserialize_codebook(). Returns 0 both when the table
   *  was built and when it was skipped; call use_precomputed_table() to tell.
   *  The table is never persisted -- rebuilding it costs
   *  nlist * dim * ksub multiply-adds, cheaper than storing it.
   */
  int build_precomputed_table(size_t max_bytes = kDefaultPrecomputeMaxBytes);

  //! Whether the L2 precomputed-table fast path is active.
  bool use_precomputed_table() const {
    return use_precomputed_table_;
  }

  //! Default cap on the precomputed table (nlist * num_chunk * ksub * 4B).
  //! Configurations above it fall back to rebuilding the LUT per list.
  static constexpr size_t kDefaultPrecomputeMaxBytes = 512UL << 20;

 private:
  //! Members
  turbo::Quantizer::Pointer quantizer_{};
  std::vector<float> centroids_{};  // nlist * dim_ (normalized if Cosine)
  //! nlist * num_chunk * ksub, only when use_precomputed_table_.
  std::vector<float> precomputed_table_{};
  size_t nlist_{0};
  size_t lut_floats_{0};  // num_chunk * ksub
  bool use_precomputed_table_{false};
  std::string quantizer_class_{};
  uint32_t dim_{0};
  uint32_t num_chunk_{0};
  bool use_zero_mean_{false};
  bool normalize_{false};  // true for Cosine metric
  bool ip_{false};         // true for InnerProduct metric (residual+dis0)
};

}  // namespace core
}  // namespace zvec
