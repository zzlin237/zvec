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

  //! Build the per-list ADC query state: fill `lut` with the quantized query
  //! LUT and set `dis0` to the per-list constant added to every ADC distance.
  //! - L2/Cosine: LUT on the residual r = (unit(q) if Cosine else q) - c,
  //!   dis0 = 0 (residual ADC directly approximates the metric distance).
  //! - IP (faiss residual+dis0): LUT on the RAW query q; the coarse center
  //!   contributes dis0 = -<q,c>, so dis0 + ADC = -<q,v>.
  void build_list_query(const void *query, size_t list_id, float *lut,
                        float *dis0) const;

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

 private:
  //! Members
  turbo::Quantizer::Pointer quantizer_{};
  std::vector<float> centroids_{};  // nlist * dim_ (normalized if Cosine)
  std::string quantizer_class_{};
  uint32_t dim_{0};
  uint32_t num_chunk_{0};
  bool use_zero_mean_{false};
  bool normalize_{false};  // true for Cosine metric
  bool ip_{false};         // true for InnerProduct metric (residual+dis0)
};

}  // namespace core
}  // namespace zvec
