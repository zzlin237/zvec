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
#include "zvec/core/framework/index_storage.h"

namespace zvec {
namespace core {

//! Segment ID used when dumping/loading the rotator data
inline const std::string RECORD_ROTATOR_SEG_ID{"integer_streaming.rotator"};

//! Rotator type exposed without rabitqlib dependency
enum class RecordRotatorType : uint8_t {
  FhtKac = 0,  //!< O(d log d) FHT-based Kac random rotation (default)
  Matrix = 1,  //!< O(d^2) explicit random matrix rotation
};

/*! RecordRotator wraps rabitqlib::Rotator for per-vector rotation.
 *
 * All rabitqlib types are hidden behind a pimpl to avoid leaking
 * rabitqlib headers to consumers of this class.
 *
 * Provides O(d log d) fast rotation (FHT-based Kac random rotation),
 * as well as serialization (save/load) of the rotation matrix.
 * Used by IntegerStreamingConverter/Reformer when enable_rotate is true.
 */
class RecordRotator {
 public:
  RecordRotator();
  ~RecordRotator();

  //! Move-only (pimpl with unique_ptr)
  RecordRotator(RecordRotator &&) noexcept;
  RecordRotator &operator=(RecordRotator &&) noexcept;
  RecordRotator(const RecordRotator &) = delete;
  RecordRotator &operator=(const RecordRotator &) = delete;

  //! Initialize the rotator
  //! @param dimension     original vector dimension
  //! @param padded_dim    padded dimension (rounded up for SIMD alignment)
  //! @param rotator_type  rotation algorithm (default: FhtKac)
  void init(size_t dimension, size_t padded_dim,
            RecordRotatorType rotator_type = RecordRotatorType::FhtKac);

  //! Rotate a single vector
  //! @param in   input vector of size >= dimension
  //! @param out  output buffer of size >= padded_dim
  void rotate(const float *in, float *out) const;

  //! Rotate a single vector into a managed buffer
  //! @param in  input vector of size >= dimension
  //! @return    vector<float> of size padded_dim containing rotated result
  std::vector<float> rotate(const float *in) const;

  //! Return the serialized size of the rotator in bytes (header + blob)
  size_t dump_bytes() const;

  //! Dump the rotator to an IndexStorage as a named segment.
  //! Same self-describing format as the dumper variant.
  int dump(const IndexStorage::Pointer &storage,
           const std::string &seg_id = RECORD_ROTATOR_SEG_ID) const;

  //! Open the rotator from an IndexStorage segment (self-describing, no init needed).
  //! Parses header to get type/dimension/padded_dim, then reconstructs the rotator.
  int open(IndexStorage::Pointer storage,
           const std::string &seg_id = RECORD_ROTATOR_SEG_ID);

  //! Load a user-specified rotation matrix.
  //! Always uses MatrixRotator internally.
  //! @param matrix       row-major matrix of shape dimension x padded_dim
  //! @param dimension    original vector dimension
  //! @param padded_dim   padded dimension (must be multiple of 64)
  int load(const float *matrix, size_t dimension, size_t padded_dim);

  //! Return the original dimension
  size_t dimension() const;

  //! Return the padded dimension
  size_t padded_dim() const;

  //! Return the rotator type
  RecordRotatorType rotator_type() const;

  //! Check if the rotator is initialized
  bool initialized() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace core
}  // namespace zvec
