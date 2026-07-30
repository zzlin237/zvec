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
#include "ivf_residual_codec.h"
#include <utility>
#include <ailego/math/normalizer.h>
#include <zvec/ailego/pattern/factory.h>
#include <zvec/core/framework/index_logger.h>

namespace zvec {
namespace core {

//! Metric names (kept local: the codec is the single owner of this policy)
static constexpr char const *kCodecL2Metric = "SquaredEuclidean";
static constexpr char const *kCodecCosineMetric = "Cosine";
static constexpr char const *kCodecIPMetric = "InnerProduct";

bool IVFResidualCodec::SupportsMetric(const std::string &metric_name) {
  return metric_name == kCodecL2Metric || metric_name == kCodecCosineMetric ||
         metric_name == kCodecIPMetric;
}

int IVFResidualCodec::init(const IndexMeta &meta,
                           const ailego::Params &params) {
  const std::string &metric = meta.metric_name();
  if (!SupportsMetric(metric)) {
    LOG_ERROR("IVFResidualCodec supports SquaredEuclidean/Cosine/InnerProduct "
              "only, metric=%s",
              metric.c_str());
    return turbo::kErrUnsupported;
  }
  normalize_ = (metric == kCodecCosineMetric);
  ip_ = (metric == kCodecIPMetric);

  dim_ = params.get_as_uint32("dim");
  if (dim_ == 0) {
    dim_ = meta.dimension();
  }
  num_chunk_ = params.get_as_uint32("num_chunk");
  if (num_chunk_ == 0) {
    num_chunk_ = 8;
  }
  if (dim_ == 0 || dim_ % num_chunk_ != 0) {
    LOG_ERROR("IVFResidualCodec: dim(%u) not divisible by num_chunk(%u)", dim_,
              num_chunk_);
    return turbo::kErrInvalidArgument;
  }
  quantizer_class_ = kDefaultQuantizerClass;
  params.get("quantizer_class", &quantizer_class_);
  if (quantizer_class_.empty()) {
    quantizer_class_ = kDefaultQuantizerClass;
  }

  //! One shared codebook (faiss-style). Encoding is always L2; the init
  //! metric only selects the search LUT: InnerProduct for IP, else L2.
  IndexMeta codec_meta = meta;
  codec_meta.set_metric(ip_ ? kCodecIPMetric : kCodecL2Metric, 0,
                        ailego::Params());
  codec_meta.set_reformer(std::string(), 0, ailego::Params());
  codec_meta.set_converter(std::string(), 0, ailego::Params());
  codec_meta.set_meta(IndexMeta::DataType::DT_FP32, dim_);

  ailego::Params quantizer_params;
  quantizer_params.set("num_chunk", static_cast<int>(num_chunk_));
  //! Forward use_zero_mean only when the caller provided it: the load path
  //! omits it so deserialize() restores the authoritative value from the
  //! codebook blob. The metric policy (zero-mean centering is invalid for
  //! InnerProduct) is decided solely by the quantizer's init().
  use_zero_mean_ = false;
  if (params.get("use_zero_mean", &use_zero_mean_)) {
    quantizer_params.set("use_zero_mean", use_zero_mean_);
  }
  int thread_count = 0;
  if (params.get("thread_count", &thread_count)) {
    quantizer_params.set("thread_count", thread_count);
  }
  //! The codec is ADC-only; skip the SDC dist_table.
  quantizer_params.set("compute_sdc", false);

  quantizer_ =
      ailego::Factory<turbo::Quantizer>::MakeShared(quantizer_class_.c_str());
  if (!quantizer_) {
    LOG_ERROR("Failed to create quantizer %s", quantizer_class_.c_str());
    return turbo::kErrInvalidArgument;
  }
  int ret = quantizer_->init(codec_meta, quantizer_params);
  if (ret != 0) {
    LOG_ERROR("Failed to init quantizer %s, ret=%d", quantizer_class_.c_str(),
              ret);
    return ret;
  }
  return 0;
}

int IVFResidualCodec::set_centroids(std::vector<float> centroids,
                                    size_t nlist) {
  if (centroids.size() != nlist * static_cast<size_t>(dim_)) {
    LOG_ERROR("IVFResidualCodec: centroids(%zu) mismatch nlist(%zu) * dim(%u)",
              centroids.size(), nlist, dim_);
    return turbo::kErrInvalidArgument;
  }
  centroids_ = std::move(centroids);
  if (normalize_) {
    for (size_t i = 0; i < nlist; ++i) {
      float norm = 0.0f;
      ailego::Normalizer<float>::L2(centroids_.data() + i * dim_, dim_, &norm);
    }
  }
  return 0;
}

void IVFResidualCodec::compute_residual(const float *vec,
                                        const float *centroid,
                                        float *out) const {
  if (normalize_) {
    for (uint32_t d = 0; d < dim_; ++d) {
      out[d] = vec[d];
    }
    float norm = 0.0f;
    ailego::Normalizer<float>::L2(out, dim_, &norm);
    for (uint32_t d = 0; d < dim_; ++d) {
      out[d] -= centroid[d];
    }
  } else {
    for (uint32_t d = 0; d < dim_; ++d) {
      out[d] = vec[d] - centroid[d];
    }
  }
}

void IVFResidualCodec::encode(const float *vec, size_t list_id,
                              uint8_t *code) const {
  thread_local std::vector<float> resid;
  resid.resize(dim_);
  this->compute_residual(vec, this->centroid(list_id), resid.data());
  quantizer_->quantize_data(resid.data(), code);
}

void IVFResidualCodec::build_list_query(const void *query, size_t list_id,
                                        float *lut, float *dis0) const {
  const float *q = reinterpret_cast<const float *>(query);
  const float *centroid = this->centroid(list_id);
  *dis0 = 0.0f;
  if (ip_) {
    quantizer_->quantize_query(q, lut);
    double dot = 0.0;
    for (uint32_t d = 0; d < dim_; ++d) {
      dot += static_cast<double>(q[d]) * centroid[d];
    }
    *dis0 = static_cast<float>(-dot);  // IP distance convention is -dot
    return;
  }
  thread_local std::vector<float> resid;
  resid.resize(dim_);
  this->compute_residual(q, centroid, resid.data());
  quantizer_->quantize_query(resid.data(), lut);
}

}  // namespace core
}  // namespace zvec
