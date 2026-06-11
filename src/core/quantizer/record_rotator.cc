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

#include "record_rotator.h"
#include <cstring>
#include <rabitqlib/utils/rotator.hpp>
#include <zvec/ailego/hash/crc32c.h>
#include "zvec/core/framework/index_error.h"
#include "zvec/core/framework/index_logger.h"

namespace zvec {
namespace core {

// All rabitqlib types are confined to this translation unit via pimpl.
struct RecordRotator::Impl {
  //! Self-describing header prepended to the rabitqlib blob on dump
  struct Header {
    uint8_t type;
    uint32_t origin_dim;
    uint32_t padded_dim;
  };

  static constexpr size_t kHeaderSize = sizeof(Header);  // 9 bytes

  size_t dimension{0};
  size_t padded_dim{0};
  RecordRotatorType type{RecordRotatorType::FhtKac};
  std::unique_ptr<rabitqlib::Rotator<float>> rotator;
  //! Inverse rotation matrix, column-major: dim x padded_dim
  //! Element [col][row] = inv_matrix_[col * dimension + row]
  //! where col in [0, padded_dim), row in [0, dimension)
  std::vector<float> inv_matrix;

  static rabitqlib::RotatorType to_rabitq(RecordRotatorType t) {
    return t == RecordRotatorType::Matrix
               ? rabitqlib::RotatorType::MatrixRotator
               : rabitqlib::RotatorType::FhtKacRotator;
  }

  static RecordRotatorType from_rabitq(uint8_t t) {
    return t == static_cast<uint8_t>(RecordRotatorType::Matrix)
               ? RecordRotatorType::Matrix
               : RecordRotatorType::FhtKac;
  }
};

RecordRotator::RecordRotator() : impl_(std::make_unique<Impl>()) {}

RecordRotator::~RecordRotator() = default;

RecordRotator::RecordRotator(RecordRotator &&) noexcept = default;
RecordRotator &RecordRotator::operator=(RecordRotator &&) noexcept = default;

void RecordRotator::init(size_t dimension, size_t padded_dim,
                         RecordRotatorType rotator_type) {
  impl_->dimension = dimension;
  impl_->padded_dim = padded_dim;
  impl_->type = rotator_type;
  impl_->rotator.reset(rabitqlib::choose_rotator<float>(
      dimension, Impl::to_rabitq(rotator_type), padded_dim));
  // Build inverse rotation data for unrotate support
  build_inverse();
}

void RecordRotator::rotate(const float *in, float *out) const {
  impl_->rotator->rotate(in, out);
}

std::vector<float> RecordRotator::rotate(const float *in) const {
  std::vector<float> out(impl_->padded_dim);
  impl_->rotator->rotate(in, out.data());
  return out;
}

void RecordRotator::build_inverse() {
  if (!impl_->rotator) {
    LOG_ERROR("RecordRotator::build_inverse: rotator not initialized");
    return;
  }

  const size_t dim = impl_->dimension;
  const size_t pdim = impl_->padded_dim;

  // Allocate column-major storage: padded_dim columns, each dim floats
  impl_->inv_matrix.resize(pdim * dim, 0.0f);

  // Compute rotation matrix by rotating each standard basis vector e_i.
  // R * e_i = i-th column of R, which we store as inv_matrix[i * dim + j].
  std::vector<float> basis(dim, 0.0f);
  std::vector<float> rotated(pdim, 0.0f);

  for (size_t i = 0; i < pdim; ++i) {
    std::fill(basis.begin(), basis.end(), 0.0f);
    if (i < dim) {
      basis[i] = 1.0f;
    }
    impl_->rotator->rotate(basis.data(), rotated.data());
    // Store as column i of the rotation matrix
    for (size_t j = 0; j < dim; ++j) {
      impl_->inv_matrix[i * dim + j] = rotated[j];
    }
  }

  LOG_DEBUG("RecordRotator::build_inverse done: dim=%zu, padded_dim=%zu",
            dim, pdim);
}

void RecordRotator::unrotate(const float *in, float *out) const {
  if (impl_->inv_matrix.empty()) {
    LOG_ERROR("RecordRotator::unrotate: build_inverse() not called");
    return;
  }

  const size_t dim = impl_->dimension;
  const size_t pdim = impl_->padded_dim;

  // Compute x = R^T * y, where y is the dim-dimensional input (padded with zeros).
  // x[j] = sum_{i=0}^{pdim-1} R[j][i] * y[i]
  //       = sum_{i=0}^{dim-1} inv_matrix_[i * dim + j] * in[i]
  // (since y[i] = 0 for i >= dim)
  std::vector<float> tmp(dim, 0.0f);
  for (size_t i = 0; i < dim; ++i) {
    const float yi = in[i];
    for (size_t j = 0; j < dim; ++j) {
      tmp[j] += impl_->inv_matrix[i * dim + j] * yi;
    }
  }
  std::memcpy(out, tmp.data(), dim * sizeof(float));
}

std::vector<float> RecordRotator::unrotate(const float *in) const {
  std::vector<float> out(impl_->dimension);
  unrotate(in, out.data());
  return out;
}

size_t RecordRotator::dump_bytes() const {
  return Impl::kHeaderSize + impl_->rotator->dump_bytes();
}

int RecordRotator::dump(const IndexStorage::Pointer &storage,
                        const std::string &seg_id) const {
  if (!storage) {
    LOG_ERROR("RecordRotator::dump(storage): null storage");
    return IndexError_InvalidArgument;
  }
  if (!impl_->rotator) {
    LOG_ERROR("RecordRotator::dump(storage): rotator not initialized");
    return IndexError_NoReady;
  }

  auto align_size = [](size_t size) -> size_t {
    return (size + 0x1F) & (~0x1F);
  };

  // Serialize: [Header: type|origin_dim|padded_dim] [rabitqlib blob]
  const size_t blob_size = impl_->rotator->dump_bytes();
  const size_t data_size = Impl::kHeaderSize + blob_size;
  const size_t total_size = align_size(data_size);
  std::vector<char> buffer(data_size);

  Impl::Header header;
  header.type = static_cast<uint8_t>(impl_->type);
  header.origin_dim = static_cast<uint32_t>(impl_->dimension);
  header.padded_dim = static_cast<uint32_t>(impl_->padded_dim);
  std::memcpy(buffer.data(), &header, Impl::kHeaderSize);
  impl_->rotator->save(buffer.data() + Impl::kHeaderSize);

  // Append segment to storage
  int ret = storage->append(seg_id, total_size);
  if (ret != 0) {
    LOG_ERROR("RecordRotator::dump(storage): append segment '%s' failed, ret=%d",
              seg_id.c_str(), ret);
    return ret;
  }

  auto segment = storage->get(seg_id);
  if (!segment) {
    LOG_ERROR("RecordRotator::dump(storage): get segment '%s' failed",
              seg_id.c_str());
    return IndexError_WriteData;
  }

  size_t written = segment->write(0, buffer.data(), data_size);
  if (written != data_size) {
    LOG_ERROR(
        "RecordRotator::dump(storage): write failed, written=%zu, expected=%zu",
        written, data_size);
    return IndexError_WriteData;
  }
  segment->resize(data_size);
  segment->update_data_crc(ailego::Crc32c::Hash(buffer.data(), data_size, 0));

  LOG_DEBUG(
      "RecordRotator::dump(storage) done: seg=%s, data_size=%zu, total=%zu",
      seg_id.c_str(), data_size, total_size);
  return 0;
}

int RecordRotator::dump(const IndexDumper::Pointer &dumper,
                        const std::string &seg_id) const {
  if (!dumper) {
    LOG_ERROR("RecordRotator::dump(dumper): null dumper");
    return IndexError_InvalidArgument;
  }
  if (!impl_->rotator) {
    LOG_ERROR("RecordRotator::dump(dumper): rotator not initialized");
    return IndexError_NoReady;
  }

  // Serialize: [Header: type|origin_dim|padded_dim] [rabitqlib blob]
  const size_t blob_size = impl_->rotator->dump_bytes();
  const size_t data_size = Impl::kHeaderSize + blob_size;
  const size_t total_size = (data_size + 0x1F) & (~0x1F);

  std::vector<char> buffer(total_size, 0);
  Impl::Header header;
  header.type = static_cast<uint8_t>(impl_->type);
  header.origin_dim = static_cast<uint32_t>(impl_->dimension);
  header.padded_dim = static_cast<uint32_t>(impl_->padded_dim);
  std::memcpy(buffer.data(), &header, Impl::kHeaderSize);
  impl_->rotator->save(buffer.data() + Impl::kHeaderSize);

  const uint32_t crc = ailego::Crc32c::Hash(buffer.data(), data_size, 0);
  const size_t padding_size = total_size - data_size;

  // Write data + padding to dumper
  if (dumper->write(buffer.data(), total_size) != total_size) {
    LOG_ERROR("RecordRotator::dump(dumper): write failed, seg=%s", seg_id.c_str());
    return IndexError_WriteData;
  }

  // Register segment
  int ret = dumper->append(seg_id, data_size, padding_size, crc);
  if (ret != 0) {
    LOG_ERROR("RecordRotator::dump(dumper): append failed, seg=%s, ret=%d",
              seg_id.c_str(), ret);
    return ret;
  }

  LOG_DEBUG(
      "RecordRotator::dump(dumper) done: seg=%s, data_size=%zu, padding=%zu",
      seg_id.c_str(), data_size, padding_size);
  return 0;
}

int RecordRotator::open(IndexStorage::Pointer storage,
                        const std::string &seg_id) {
  if (!storage) {
    LOG_ERROR("RecordRotator::open: null storage");
    return IndexError_InvalidArgument;
  }

  auto segment = storage->get(seg_id);
  if (!segment) {
    LOG_ERROR("RecordRotator::open: segment '%s' not found", seg_id.c_str());
    return IndexError_InvalidFormat;
  }

  // Read the rotator data from the segment (header + blob)
  const size_t data_size = segment->data_size();
  if (data_size <= Impl::kHeaderSize) {
    LOG_ERROR("RecordRotator::open: data too small (%zu bytes)", data_size);
    return IndexError_InvalidFormat;
  }

  IndexStorage::MemoryBlock block;
  size_t read_size = segment->read(0, block, data_size);
  if (read_size != data_size) {
    LOG_ERROR("RecordRotator::open: read failed, read=%zu, expected=%zu",
              read_size, data_size);
    return IndexError_InvalidFormat;
  }

  // Verify CRC if available (covers header + blob)
  uint32_t expected_crc = segment->data_crc();
  if (expected_crc != 0) {
    uint32_t actual_crc = ailego::Crc32c::Hash(block.data(), data_size, 0);
    if (actual_crc != expected_crc) {
      LOG_ERROR(
          "RecordRotator::open: CRC mismatch, expected=0x%08x, actual=0x%08x",
          expected_crc, actual_crc);
      return IndexError_InvalidFormat;
    }
  }

  // Parse self-describing header
  const char *raw = reinterpret_cast<const char *>(block.data());
  Impl::Header header;
  std::memcpy(&header, raw, Impl::kHeaderSize);

  impl_->type = Impl::from_rabitq(header.type);
  impl_->dimension = static_cast<size_t>(header.origin_dim);
  impl_->padded_dim = static_cast<size_t>(header.padded_dim);

  // Reconstruct the rotator from header info and load blob
  impl_->rotator.reset(rabitqlib::choose_rotator<float>(
      impl_->dimension, Impl::to_rabitq(impl_->type), impl_->padded_dim));
  impl_->rotator->load(raw + Impl::kHeaderSize);

  LOG_DEBUG(
      "RecordRotator::open done: seg=%s, dim=%zu, padded_dim=%zu, "
      "data_size=%zu",
      seg_id.c_str(), impl_->dimension, impl_->padded_dim, data_size);

  // Build inverse rotation data for unrotate support
  build_inverse();

  return 0;
}

int RecordRotator::load(const float *matrix, size_t dimension,
                        size_t padded_dim) {
  if (!matrix) {
    LOG_ERROR("RecordRotator::load: null matrix");
    return IndexError_InvalidArgument;
  }
  if (dimension == 0 || padded_dim == 0) {
    LOG_ERROR("RecordRotator::load: invalid dims %zu x %zu", dimension,
              padded_dim);
    return IndexError_InvalidArgument;
  }

  impl_->dimension = dimension;
  impl_->padded_dim = padded_dim;
  impl_->type = RecordRotatorType::Matrix;
  impl_->rotator.reset(rabitqlib::choose_rotator<float>(
      dimension, rabitqlib::RotatorType::MatrixRotator, padded_dim));
  impl_->rotator->load(reinterpret_cast<const char *>(matrix));

  LOG_DEBUG("RecordRotator::load done: dim=%zu, padded_dim=%zu",
            dimension, padded_dim);

  // Build inverse rotation data for unrotate support
  build_inverse();

  return 0;
}

size_t RecordRotator::dimension() const {
  return impl_->dimension;
}

size_t RecordRotator::padded_dim() const {
  return impl_->padded_dim;
}

RecordRotatorType RecordRotator::rotator_type() const {
  return impl_->type;
}

bool RecordRotator::initialized() const {
  return impl_->rotator != nullptr;
}

}  // namespace core
}  // namespace zvec
