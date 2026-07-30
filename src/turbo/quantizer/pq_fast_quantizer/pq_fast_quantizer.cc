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

#include "quantizer/pq_fast_quantizer/pq_fast_quantizer.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <limits>
#include <random>
#include <vector>
#include <ailego/algorithm/kmeans.h>
#include <zvec/core/framework/index_factory.h>
#include <zvec/core/framework/index_threads.h>
#include "common/fast_scan_common.h"

namespace zvec {
namespace turbo {

// ---------------------------------------------------------------------------
// FastScan PQ serialization payload (follows the QuantizerSerHeader).
// ---------------------------------------------------------------------------
struct PqFastSerPayload {
  uint32_t original_dim;
  uint32_t num_chunk;
  uint32_t sub_dim;
  uint32_t num_centroids;  // always 16
  uint8_t use_zero_mean;
  uint8_t input_data_type;  // turbo DataType: kFp32=3, kFp16=2
  uint8_t reserved[2];
};

void PqFastQuantizer::setup_functions() {
  // ISA-dispatched FastScan block kernel.
  scan_fn_ = get_pq_kernels(DataType::kInt4, QuantizeType::kPQFast).fast_scan;

  // L2-only batch distance for encoding and KMeans training: the PQ
  // codebook is trained/encoded in L2 space regardless of the search metric.
  l2_batch_fn_ =
      get_batch_distance_func(MetricType::kSquaredEuclidean, input_data_type_,
                              QuantizeType::kDefault, CpuArchType::kAuto);

  // Metric-aware batch distance for the search LUT (L2 or IP only).
  batch_fn_ = get_batch_distance_func(metric_from_name(meta_.metric_name()),
                                      input_data_type_, QuantizeType::kDefault,
                                      CpuArchType::kAuto);
}

int PqFastQuantizer::init(const IndexMeta &meta, const ailego::Params &params) {
  meta_ = meta;

  // Map core IndexMeta::DataType to turbo DataType.
  if (meta.data_type() == IndexMeta::DataType::DT_FP16) {
    input_data_type_ = DataType::kFp16;
  } else if (meta.data_type() == IndexMeta::DataType::DT_FP32) {
    input_data_type_ = DataType::kFp32;
  } else {
    return kErrUnsupported;
  }

  // FastScan is a batch-scan quantizer for L2 / IP only.  Cosine callers
  // (e.g. IVF residual PQ) normalize upfront and search with L2.
  const auto &metric_name = meta_.metric_name();
  if (metric_name != "SquaredEuclidean" && metric_name != "InnerProduct") {
    return kErrUnsupported;
  }

  uint32_t d = meta.dimension();
  original_dim_ = d;

  // Read num_chunk from params (required).
  uint32_t nsq = 0;
  if (!params.get("num_chunk", &nsq) || nsq == 0) {
    return kErrUnsupported;
  }
  if (d % nsq != 0) {
    return kErrUnsupported;
  }

  num_chunk_ = nsq;
  sub_dim_ = d / nsq;

  // Pre-allocate centroids as byte buffer (filled by train()).
  centroids_.resize(static_cast<size_t>(num_chunk_) * kNumCentroids * sub_dim_ *
                    element_size());

  setup_functions();

  // Read optional training params (aligned with multi_chunk_cluster)
  params.get("thread_count", &thread_count_);
  params.get("markov_chain_length", &markov_chain_length_);
  params.get("epsilon", &epsilon_);
  params.get("use_zero_mean", &use_zero_mean_);

  // Zero-mean centering shifts the space, which breaks the IP LUT
  // decomposition -- keep it for L2 only.
  if (use_zero_mean_ && metric_name != "SquaredEuclidean") {
    use_zero_mean_ = false;
  }

  meta_.set_meta(IndexMeta::DataType::DT_INT4, num_chunk_);
  return 0;
}

// ---------------------------------------------------------------------------
// Training (identical paradigm to PqInt4Quantizer: L2 KMeans, k=16).
// ---------------------------------------------------------------------------

void PqFastQuantizer::build_centroid_ptrs_cache() {
  const size_t k = kNumCentroids;
  const size_t d = sub_dim_;
  const uint8_t *base = centroids_.data();
  const size_t type_sz = element_size();
  centroid_ptrs_cache_.resize(num_chunk_);
  for (uint32_t m = 0; m < num_chunk_; ++m) {
    auto &ptrs = centroid_ptrs_cache_[m];
    ptrs.resize(k);
    for (size_t c = 0; c < k; ++c) {
      ptrs[c] = base + (static_cast<size_t>(m) * k * d + c * d) * type_sz;
    }
  }
}

template <typename T>
void PqFastQuantizer::train_subquantizer(const T *data, size_t num,
                                         size_t stride, size_t sub_idx) {
  const size_t k = kNumCentroids;
  const size_t d = sub_dim_;
  uint8_t *centroids_m =
      centroids_.data() + static_cast<size_t>(sub_idx) * k * d * sizeof(T);

  // Non-spherical L2 KMeans: the PQ codebook must minimize L2 reconstruction
  // error, so centroids are the true (magnitude-preserving) means.
  ailego::NumericalKmeans<T, SingleQueueIndexThreads> algorithm(k, d);

  // Append sub-vectors (NumericalKmeans handles transpose internally)
  for (size_t i = 0; i < num; ++i) {
    const T *sub_vec =
        reinterpret_cast<const T *>(reinterpret_cast<const uint8_t *>(data) +
                                    i * stride) +
        sub_idx * d;
    algorithm.append(sub_vec, d);
  }

  // Single-threaded pool — parallelism is at the sub-quantizer level.
  auto local_threads = std::make_shared<SingleQueueIndexThreads>(1, false);

  // KMC2 centroid initialization.
  ailego::Kmc2CentroidsGenerator<
      ailego::NumericalKmeans<T, SingleQueueIndexThreads>,
      SingleQueueIndexThreads>
      gen;
  gen.set_chain_length(markov_chain_length_);
  gen.set_assumption_free(false);
  algorithm.init_centroids(*local_threads, gen);

  // Lloyd iterations
  double cost = 0.0;
  for (uint32_t iter = 0; iter < kMaxKmeansIters; ++iter) {
    double old_cost = cost;
    bool result = algorithm.cluster_once(*local_threads, &cost);
    if (!result) {
      break;
    }
    double new_epsilon = std::abs(cost - old_cost);
    if (new_epsilon < epsilon_) {
      break;
    }
  }

  // Extract centroids into the flat centroids_ byte buffer
  const auto &cents = algorithm.centroids();
  for (size_t c = 0; c < cents.count(); ++c) {
    std::memcpy(centroids_m + c * d * sizeof(T), cents[c], d * sizeof(T));
  }
}

int PqFastQuantizer::train(IndexHolder::Pointer holder) {
  return train(holder, static_cast<int>(thread_count_));
}

int PqFastQuantizer::train(IndexHolder::Pointer holder, int thread_count) {
  if (!holder) {
    return kErrUnsupported;
  }

  size_t num = holder->count();
  const uint32_t elem_sz = element_size();

  // Collect all data into a contiguous byte buffer (original data type).
  auto iter = holder->create_iterator();
  std::vector<uint8_t> all_data(num * original_dim_ * elem_sz);
  size_t row = 0;
  for (; iter->is_valid(); iter->next(), ++row) {
    std::memcpy(all_data.data() + row * original_dim_ * elem_sz, iter->data(),
                original_dim_ * elem_sz);
  }

  // Subsample if the dataset exceeds the training limit.
  if (num > kMaxTrainVectors) {
    std::mt19937 rng(42);
    // Fisher-Yates partial shuffle: randomly place kMaxTrainVectors vectors
    // at the front of the buffer.
    for (size_t i = 0; i < kMaxTrainVectors; ++i) {
      std::uniform_int_distribution<size_t> dist(i, num - 1);
      size_t j = dist(rng);
      if (i != j) {
        // Swap full vectors (dim-sized chunks in bytes).
        size_t vec_bytes = original_dim_ * elem_sz;
        for (size_t b = 0; b < vec_bytes; ++b) {
          std::swap(all_data[i * vec_bytes + b], all_data[j * vec_bytes + b]);
        }
      }
    }
    num = kMaxTrainVectors;
    all_data.resize(num * original_dim_ * elem_sz);
    all_data.shrink_to_fit();
  }

  size_t data_stride = original_dim_ * elem_sz;

  // Zero-mean centering: subtract the per-dimension mean; the centroid is
  // saved for quantize_data / quantize_query / dequantize.
  if (use_zero_mean_) {
    switch (input_data_type_) {
      case DataType::kFp16:
        compute_and_subtract_center(
            reinterpret_cast<ailego::Float16 *>(all_data.data()), num);
        break;
      case DataType::kFp32:
        compute_and_subtract_center(reinterpret_cast<float *>(all_data.data()),
                                    num);
        break;
      default:
        break;
    }
  }

  // Create thread pool.
  auto threads = std::make_shared<SingleQueueIndexThreads>(
      static_cast<uint32_t>(thread_count), false);
  auto task_group = threads->make_group();

  // Distribute sub-quantizers across threads.
  std::atomic<size_t> finished{0};
  size_t pool_count = threads->count();

  auto submit_training = [&](const auto *typed_data) {
    using T = std::remove_const_t<std::remove_pointer_t<decltype(typed_data)>>;
    for (size_t i = 0; i < pool_count; ++i) {
      task_group->submit(ailego::Closure::New(
          [this, typed_data, num, data_stride, i, pool_count, &finished]() {
            for (uint32_t m = static_cast<uint32_t>(i); m < num_chunk_;
                 m += static_cast<uint32_t>(pool_count)) {
              train_subquantizer<T>(typed_data, num, data_stride, m);
              finished++;
            }
          }));
    }
  };

  switch (input_data_type_) {
    case DataType::kFp16:
      submit_training(
          reinterpret_cast<const ailego::Float16 *>(all_data.data()));
      break;
    case DataType::kFp32:
      submit_training(reinterpret_cast<const float *>(all_data.data()));
      break;
    default:
      break;
  }
  task_group->wait_finish();

  // Pre-build centroid pointer cache for fast encode/search.
  build_centroid_ptrs_cache();
  return 0;
}

// ---------------------------------------------------------------------------
// Encoding
// ---------------------------------------------------------------------------

void PqFastQuantizer::quantize_data(const void *input, void *output) const {
  uint8_t *code = reinterpret_cast<uint8_t *>(output);
  const uint32_t elem_sz = element_size();
  const void *vec = input;

  // Zero-mean centering: subtract centroid before encoding.
  std::vector<uint8_t> centered_vec_storage;
  if (use_zero_mean_) {
    centered_vec_storage.resize(original_dim_ * elem_sz);
    std::memcpy(centered_vec_storage.data(), vec, original_dim_ * elem_sz);
    switch (input_data_type_) {
      case DataType::kFp16:
        subtract_center(
            reinterpret_cast<ailego::Float16 *>(centered_vec_storage.data()));
        break;
      case DataType::kFp32:
        subtract_center(reinterpret_cast<float *>(centered_vec_storage.data()));
        break;
      default:
        break;
    }
    vec = centered_vec_storage.data();
  }

  // Zero the packed code buffer first: nibble packing ORs codes in, and an
  // odd num_chunk leaves the last byte's high nibble as the (zero) pad.
  const size_t packed_len = packed_code_length();
  std::memset(code, 0, packed_len);

  // Encode with L2-only batch distance (search-metric independent),
  // fusing argmin into the distance loop.
  float dists[kNumCentroids];
  const uint8_t *vec_bytes = reinterpret_cast<const uint8_t *>(vec);

  for (uint32_t m = 0; m < num_chunk_; ++m) {
    const void *sub_vec =
        vec_bytes + static_cast<size_t>(m) * sub_dim_ * elem_sz;
    const auto &centroid_ptrs = centroid_ptrs_cache_[m];

    // Compute L2 distances from this sub-vector to all 16 centroids.
    // const_cast: .data() returns const void* const* but the kernel expects
    // const void** and never modifies the pointer array.
    l2_batch_fn_(const_cast<const void **>(centroid_ptrs.data()), sub_vec,
                 kNumCentroids, sub_dim_, dists);

    // Argmin: find nearest centroid.
    float best_dist = dists[0];
    uint32_t best_idx = 0;
    for (uint32_t j = 1; j < kNumCentroids; ++j) {
      if (dists[j] < best_dist) {
        best_dist = dists[j];
        best_idx = j;
      }
    }

    // Pack the 4-bit code: low nibble for even m, high nibble for odd m.
    code[m >> 1] |= static_cast<uint8_t>((m & 1) ? (best_idx << 4) : best_idx);
  }
}

void PqFastQuantizer::compute_float_lut(const void *input, float *lut) const {
  const uint32_t elem_sz = element_size();
  const void *query = input;

  // Zero-mean centering: subtract centroid before LUT computation.
  std::vector<uint8_t> centered_query_storage;
  if (use_zero_mean_) {
    centered_query_storage.resize(original_dim_ * elem_sz);
    std::memcpy(centered_query_storage.data(), query, original_dim_ * elem_sz);
    switch (input_data_type_) {
      case DataType::kFp16:
        subtract_center(
            reinterpret_cast<ailego::Float16 *>(centered_query_storage.data()));
        break;
      case DataType::kFp32:
        subtract_center(
            reinterpret_cast<float *>(centered_query_storage.data()));
        break;
      default:
        break;
    }
    query = centered_query_storage.data();
  }

  // LUT[m][j] = distance(q_m, c_m[j]) via the metric-aware batch_fn_:
  // L2: ||q_m - c_m[j]||^2   IP: -dot(q_m, c_m[j]).
  const uint8_t *query_bytes = reinterpret_cast<const uint8_t *>(query);
  for (uint32_t m = 0; m < num_chunk_; ++m) {
    const auto &centroid_ptrs = centroid_ptrs_cache_[m];
    const void *sub_query =
        query_bytes + static_cast<size_t>(m) * sub_dim_ * elem_sz;
    batch_fn_(const_cast<const void **>(centroid_ptrs.data()), sub_query,
              kNumCentroids, sub_dim_, lut + m * kNumCentroids);
  }
}

size_t PqFastQuantizer::quantized_query_vector_length() const {
  return fast_scan_packed_lut_size(num_chunk_) + 2 * sizeof(float);
}

void PqFastQuantizer::quantize_query(const void *input, void *output) const {
  const size_t lut_size = static_cast<size_t>(num_chunk_) * kNumCentroids;
  std::vector<float> lut(lut_size);
  compute_float_lut(input, lut.data());

  // Affine-quantize the whole table with a single min/max so that the
  // integer accumulation keeps the correct additive semantics.
  float lo = lut[0];
  float hi = lut[0];
  for (size_t i = 1; i < lut_size; ++i) {
    lo = std::min(lo, lut[i]);
    hi = std::max(hi, lut[i]);
  }
  const float delta = (hi - lo) / 255.0f;
  const float inv_delta = (delta > 0.0f) ? (1.0f / delta) : 0.0f;

  std::vector<uint8_t> u8_lut(lut_size);
  for (size_t i = 0; i < lut_size; ++i) {
    float q = std::round((lut[i] - lo) * inv_delta);
    u8_lut[i] = static_cast<uint8_t>(std::min(std::max(q, 0.0f), 255.0f));
  }

  // [packed u8 LUT | float delta | float bias]
  uint8_t *out = reinterpret_cast<uint8_t *>(output);
  fast_scan_pack_lut(u8_lut.data(), num_chunk_, out);
  const float bias = static_cast<float>(num_chunk_) * lo;
  std::memcpy(out + fast_scan_packed_lut_size(num_chunk_), &delta,
              sizeof(float));
  std::memcpy(out + fast_scan_packed_lut_size(num_chunk_) + sizeof(float),
              &bias, sizeof(float));
}

int PqFastQuantizer::pack_codes(const void *codes, size_t num, size_t stride,
                                void *out) const {
  if (!codes || !out) {
    return kErrInvalidArgument;
  }
  fast_scan_pack_codes(reinterpret_cast<const uint8_t *>(codes), num, stride,
                       num_chunk_, reinterpret_cast<uint8_t *>(out));
  return 0;
}

// ---------------------------------------------------------------------------
// Distance computation
// ---------------------------------------------------------------------------

float PqFastQuantizer::adc_u8(const uint8_t *code,
                              const uint8_t *qquery) const {
  // The packed LUT keeps sub-space m at offset m * 16, so a plain code can
  // be looked up directly.
  int32_t accu = 0;
  for (uint32_t m = 0; m < num_chunk_; ++m) {
    const uint8_t c =
        static_cast<uint8_t>(code[m >> 1] >> ((m & 1) * 4)) & 0x0F;
    accu += qquery[static_cast<size_t>(m) * 16 + c];
  }
  float delta = 0.0f;
  float bias = 0.0f;
  const uint8_t *tail = qquery + fast_scan_packed_lut_size(num_chunk_);
  std::memcpy(&delta, tail, sizeof(float));
  std::memcpy(&bias, tail + sizeof(float), sizeof(float));
  return static_cast<float>(accu) * delta + bias;
}

float PqFastQuantizer::calc_distance_dp_query(const void *dp,
                                              const void *query) const {
  // dp = plain nibble-packed pq_code; query = quantized query (u8 LUT).
  return adc_u8(reinterpret_cast<const uint8_t *>(dp),
                reinterpret_cast<const uint8_t *>(query));
}

void PqFastQuantizer::calc_distance_dp_query_batch(const void *const *dp_list,
                                                   int dp_num,
                                                   const void *query,
                                                   float *dist_list) const {
  const uint8_t *q = reinterpret_cast<const uint8_t *>(query);
  for (int i = 0; i < dp_num; ++i) {
    dist_list[i] = adc_u8(reinterpret_cast<const uint8_t *>(dp_list[i]), q);
  }
}

void PqFastQuantizer::calc_distance_dp_query_batch_contiguous(
    const void *codes, int dp_num, size_t stride, const void *query,
    float *dist_list) const {
  // `codes` is one (or several back-to-back) packed 32-vector block(s)
  // produced by pack_codes(); the per-vector stride is meaningless in the
  // packed layout.
  (void)stride;
  const uint8_t *q = reinterpret_cast<const uint8_t *>(query);
  float delta = 0.0f;
  float bias = 0.0f;
  const uint8_t *tail = q + fast_scan_packed_lut_size(num_chunk_);
  std::memcpy(&delta, tail, sizeof(float));
  std::memcpy(&bias, tail + sizeof(float), sizeof(float));

  const uint8_t *block = reinterpret_cast<const uint8_t *>(codes);
  const size_t block_bytes = fast_scan_packed_block_size(num_chunk_);
  int32_t accu32[kFastScanBlockSize];
  int done = 0;
  while (done < dp_num) {
    const int n = std::min(static_cast<int>(kFastScanBlockSize), dp_num - done);
    scan_fn_(block, q, num_chunk_, accu32);
    for (int i = 0; i < n; ++i) {
      dist_list[done + i] = static_cast<float>(accu32[i]) * delta + bias;
    }
    block += block_bytes;
    done += n;
  }
}

float PqFastQuantizer::calc_distance_dp_query_unquantized(
    const void *dp, const void *query) const {
  // Exact float ADC over the on-the-fly LUT (no uint8 quantization).
  std::vector<float> lut(static_cast<size_t>(num_chunk_) * kNumCentroids);
  compute_float_lut(query, lut.data());
  const uint8_t *code = reinterpret_cast<const uint8_t *>(dp);
  float d = 0.0f;
  for (uint32_t m = 0; m < num_chunk_; ++m) {
    const uint8_t c =
        static_cast<uint8_t>(code[m >> 1] >> ((m & 1) * 4)) & 0x0F;
    d += lut[static_cast<size_t>(m) * kNumCentroids + c];
  }
  return d;
}

void PqFastQuantizer::calc_distance_dp_query_batch_unquantized(
    const void *const *dp_list, int dp_num, const void *query,
    float *dist_list) const {
  std::vector<float> lut(static_cast<size_t>(num_chunk_) * kNumCentroids);
  compute_float_lut(query, lut.data());
  for (int i = 0; i < dp_num; ++i) {
    const uint8_t *code = reinterpret_cast<const uint8_t *>(dp_list[i]);
    float d = 0.0f;
    for (uint32_t m = 0; m < num_chunk_; ++m) {
      const uint8_t c =
          static_cast<uint8_t>(code[m >> 1] >> ((m & 1) * 4)) & 0x0F;
      d += lut[static_cast<size_t>(m) * kNumCentroids + c];
    }
    dist_list[i] = d;
  }
}

float PqFastQuantizer::calc_distance_dp_dp(const void *dp1,
                                           const void *dp2) const {
  // FastScan is a pure batch-scan quantizer: no SDC dist_table is kept.
  (void)dp1;
  (void)dp2;
  return 0.0f;
}

int PqFastQuantizer::quantize(const void *query, const IndexQueryMeta &qmeta,
                              std::string *out, IndexQueryMeta *ometa) const {
  // Validate unit_size against the input data type.
  size_t expected_unit = 0;
  switch (input_data_type_) {
    case DataType::kFp16:
      expected_unit = sizeof(ailego::Float16);
      break;
    case DataType::kFp32:
      expected_unit = sizeof(float);
      break;
    default:
      break;
  }
  if (qmeta.unit_size() != expected_unit) {
    return kErrUnsupported;
  }

  size_t lut_bytes = quantized_query_vector_length();
  out->resize(lut_bytes);
  quantize_query(query, &(*out)[0]);

  *ometa = qmeta;
  ometa->set_meta(IndexMeta::DataType::DT_INT4, num_chunk_,
                  static_cast<uint32_t>(type_), 0);
  return 0;
}

int PqFastQuantizer::dequantize(const void *in, const IndexQueryMeta &qmeta,
                                std::string *out) const {
  (void)qmeta;
  const uint8_t *code = reinterpret_cast<const uint8_t *>(in);
  size_t byte_size = static_cast<size_t>(original_dim_) * sizeof(float);
  out->resize(byte_size);
  float *result = reinterpret_cast<float *>(&(*out)[0]);

  // Reconstruct by concatenating the selected centroids per sub-quantizer.
  const size_t k = kNumCentroids;
  const size_t d = sub_dim_;
  const uint32_t elem_sz = element_size();
  for (uint32_t m = 0; m < num_chunk_; ++m) {
    const uint8_t *centroids_m =
        centroids_.data() + static_cast<size_t>(m) * k * d * elem_sz;
    // Unpack the 4-bit code: low nibble for even m, high nibble for odd m.
    uint8_t c = static_cast<uint8_t>((code[m >> 1] >> ((m & 1) * 4)) & 0x0F);
    const uint8_t *centroid =
        centroids_m + static_cast<size_t>(c) * d * elem_sz;
    switch (input_data_type_) {
      case DataType::kFp16: {
        const ailego::Float16 *src =
            reinterpret_cast<const ailego::Float16 *>(centroid);
        for (size_t j = 0; j < d; ++j) {
          result[m * d + j] = static_cast<float>(src[j]);
        }
        break;
      }
      case DataType::kFp32:
        std::memcpy(result + m * d, centroid, d * sizeof(float));
        break;
      default:
        break;
    }
  }

  // Undo zero-mean centering: add the centroid back.
  if (use_zero_mean_) {
    for (uint32_t j = 0; j < original_dim_; ++j) {
      result[j] += centroid_[j];
    }
  }
  return 0;
}

// ---------------------------------------------------------------------------
// Serialization
// ---------------------------------------------------------------------------
int PqFastQuantizer::serialize(std::string *out) const {
  if (!out) return kErrUnsupported;

  QuantizerSerHeader hdr{};
  hdr.magic = kQuantizerMagic;
  hdr.version = kQuantizerSerVersion;
  hdr.quant_type = static_cast<uint32_t>(QuantizeType::kPQFast);
  hdr.dim = original_dim_;
  hdr.metric = static_cast<uint32_t>(metric_from_name(meta_.metric_name()));

  PqFastSerPayload payload{};
  payload.original_dim = original_dim_;
  payload.num_chunk = num_chunk_;
  payload.sub_dim = sub_dim_;
  payload.num_centroids = kNumCentroids;
  payload.use_zero_mean = use_zero_mean_ ? 1 : 0;
  payload.input_data_type = static_cast<uint8_t>(input_data_type_);

  size_t centroids_bytes = centroids_.size();  // already byte buffer
  size_t centroid_bytes = use_zero_mean_ ? centroid_.size() * sizeof(float) : 0;
  hdr.payload_size =
      static_cast<uint32_t>(sizeof(payload) + centroids_bytes + centroid_bytes);

  out->clear();
  out->append(reinterpret_cast<const char *>(&hdr), sizeof(hdr));
  out->append(reinterpret_cast<const char *>(&payload), sizeof(payload));
  out->append(reinterpret_cast<const char *>(centroids_.data()),
              centroids_bytes);
  // Append zero-mean centroid for centering support.
  if (use_zero_mean_) {
    out->append(reinterpret_cast<const char *>(centroid_.data()),
                centroid_bytes);
  }
  return 0;
}

int PqFastQuantizer::deserialize(std::string &in) {
  return deserialize(in.data(), in.size());
}

int PqFastQuantizer::deserialize(const void *data, size_t len) {
  if (len < sizeof(QuantizerSerHeader) + sizeof(PqFastSerPayload)) {
    return kErrUnsupported;
  }

  const char *ptr = reinterpret_cast<const char *>(data);
  QuantizerSerHeader hdr;
  std::memcpy(&hdr, ptr, sizeof(hdr));
  ptr += sizeof(hdr);

  if (hdr.magic != kQuantizerMagic) return kErrUnsupported;
  if (hdr.quant_type != static_cast<uint16_t>(QuantizeType::kPQFast)) {
    return kErrUnsupported;
  }

  PqFastSerPayload payload;
  std::memcpy(&payload, ptr, sizeof(payload));
  ptr += sizeof(payload);

  original_dim_ = payload.original_dim;
  num_chunk_ = payload.num_chunk;
  sub_dim_ = payload.sub_dim;
  input_data_type_ = static_cast<DataType>(payload.input_data_type);
  if (input_data_type_ != DataType::kFp32 &&
      input_data_type_ != DataType::kFp16) {
    return kErrUnsupported;
  }

  // Restore centroids (raw bytes in original data type).
  size_t centroids_bytes = static_cast<size_t>(num_chunk_) * kNumCentroids *
                           sub_dim_ * element_size();
  centroids_.resize(centroids_bytes);
  std::memcpy(centroids_.data(), ptr, centroids_bytes);
  ptr += centroids_bytes;

  // Restore zero-mean centroid if centering was enabled.
  if (payload.use_zero_mean) {
    use_zero_mean_ = true;
    size_t centroid_bytes = static_cast<size_t>(original_dim_) * sizeof(float);
    centroid_.resize(original_dim_);
    std::memcpy(centroid_.data(), ptr, centroid_bytes);
    ptr += centroid_bytes;
  }

  // Re-dispatch kernels and batch distance functions.
  setup_functions();

  // Set output meta: the quantized representation is INT4 codes.
  meta_.set_meta(IndexMeta::DataType::DT_INT4, num_chunk_);

  // Pre-build centroid pointer cache for fast encode/search.
  build_centroid_ptrs_cache();

  return 0;
}

INDEX_FACTORY_REGISTER_QUANTIZER(PqFastQuantizer);

// ---------------------------------------------------------------------------
// Template helper implementations (type-dispatched at call sites)
// ---------------------------------------------------------------------------

template <typename T>
void PqFastQuantizer::compute_and_subtract_center(T *data, size_t num) {
  centroid_.assign(original_dim_, 0.0f);
  for (size_t i = 0; i < num; ++i) {
    const T *v = data + i * original_dim_;
    for (uint32_t d = 0; d < original_dim_; ++d) {
      centroid_[d] += static_cast<float>(v[d]);
    }
  }
  for (uint32_t d = 0; d < original_dim_; ++d) {
    centroid_[d] /= static_cast<float>(num);
  }
  for (size_t i = 0; i < num; ++i) {
    T *v = data + i * original_dim_;
    for (uint32_t d = 0; d < original_dim_; ++d) {
      v[d] -= centroid_[d];
    }
  }
}

template <typename T>
void PqFastQuantizer::subtract_center(T *vec) const {
  for (uint32_t d = 0; d < original_dim_; ++d) {
    vec[d] -= centroid_[d];
  }
}

}  // namespace turbo
}  // namespace zvec
