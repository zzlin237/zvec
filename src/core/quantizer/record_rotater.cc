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

#include "record_rotater.h"
#include <cstring>
#include <rabitqlib/utils/rotator.hpp>
#include <zvec/ailego/hash/crc32c.h>
#include "zvec/core/framework/index_error.h"
#include "zvec/core/framework/index_logger.h"

namespace zvec {
namespace core {

// All rabitqlib types are confined to this translation unit via pimpl.
struct RecordRotator::Impl {
  size_t dimension{0};
  size_t padded_dim{0};
  RecordRotatorType type{RecordRotatorType::FhtKac};
  std::unique_ptr<rabitqlib::Rotator<float>> rotator;

  static rabitqlib::RotatorType to_rabitq(RecordRotatorType t) {
    return t == RecordRotatorType::Matrix
               ? rabitqlib::RotatorType::MatrixRotator
               : rabitqlib::RotatorType::FhtKacRotator;
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
}

void RecordRotator::rotate(const float *in, float *out) const {
  impl_->rotator->rotate(in, out);
}

std::vector<float> RecordRotator::rotate(const float *in) const {
  std::vector<float> out(impl_->padded_dim);
  impl_->rotator->rotate(in, out.data());
  return out;
}

size_t RecordRotator::dump_bytes() const {
  return impl_->rotator->dump_bytes();
}

int RecordRotator::dump(const IndexDumper::Pointer &dumper,
                        const std::string &seg_id) const {
  if (!dumper) {
    LOG_ERROR("RecordRotator::dump: null dumper");
    return IndexError_InvalidArgument;
  }
  if (!impl_->rotator) {
    LOG_ERROR("RecordRotator::dump: rotator not initialized");
    return IndexError_NoReady;
  }

  auto align_size = [](size_t size) -> size_t {
    return (size + 0x1F) & (~0x1F);
  };

  // Serialize rotator to buffer
  const size_t data_size = impl_->rotator->dump_bytes();
  std::vector<char> buffer(data_size);
  impl_->rotator->save(buffer.data());

  // Write rotator data
  size_t written = dumper->write(buffer.data(), data_size);
  if (written != data_size) {
    LOG_ERROR("RecordRotator::dump: write failed, written=%zu, expected=%zu",
              written, data_size);
    return IndexError_WriteData;
  }
  uint32_t crc = ailego::Crc32c::Hash(buffer.data(), data_size, 0);

  // Write padding for 32-byte alignment
  size_t padding_size = align_size(data_size) - data_size;
  if (padding_size > 0) {
    std::string padding(padding_size, '\0');
    if (dumper->write(padding.data(), padding_size) != padding_size) {
      LOG_ERROR("RecordRotator::dump: padding write failed");
      return IndexError_WriteData;
    }
  }

  // Register segment meta
  int ret = dumper->append(seg_id, data_size, padding_size, crc);
  if (ret != 0) {
    LOG_ERROR("RecordRotator::dump: append segment meta failed, ret=%d", ret);
    return ret;
  }

  LOG_DEBUG("RecordRotator::dump done: seg=%s, data_size=%zu, padding=%zu",
            seg_id.c_str(), data_size, padding_size);
  return 0;
}

int RecordRotator::load(IndexStorage::Pointer storage,
                        const std::string &seg_id) {
  if (!storage) {
    LOG_ERROR("RecordRotator::load: null storage");
    return IndexError_InvalidArgument;
  }

  auto segment = storage->get(seg_id);
  if (!segment) {
    LOG_ERROR("RecordRotator::load: segment '%s' not found", seg_id.c_str());
    return IndexError_InvalidFormat;
  }

  // Read the rotator data from the segment
  const size_t data_size = segment->data_size();
  IndexStorage::MemoryBlock block;
  size_t read_size = segment->read(0, block, data_size);
  if (read_size != data_size) {
    LOG_ERROR("RecordRotator::load: read failed, read=%zu, expected=%zu",
              read_size, data_size);
    return IndexError_InvalidFormat;
  }

  // Verify CRC if available
  uint32_t expected_crc = segment->data_crc();
  if (expected_crc != 0) {
    uint32_t actual_crc = ailego::Crc32c::Hash(block.data(), data_size, 0);
    if (actual_crc != expected_crc) {
      LOG_ERROR(
          "RecordRotator::load: CRC mismatch, expected=0x%08x, actual=0x%08x",
          expected_crc, actual_crc);
      return IndexError_InvalidFormat;
    }
  }

  // Reconstruct the rotator from serialized data
  impl_->rotator.reset(rabitqlib::choose_rotator<float>(
      impl_->dimension, Impl::to_rabitq(impl_->type), impl_->padded_dim));
  impl_->rotator->load(reinterpret_cast<const char *>(block.data()));

  LOG_DEBUG(
      "RecordRotator::load done: seg=%s, dim=%zu, padded_dim=%zu, "
      "data_size=%zu",
      seg_id.c_str(), impl_->dimension, impl_->padded_dim, data_size);
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
