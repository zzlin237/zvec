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

// Quantized graph (QG) support for the HNSW graph: the scheme NGT-QG
// introduced, where a node record also carries the quantized codes of all its
// level 0 neighbors.  A graph hop then reads one contiguous region and scans
// it in a single batch, instead of chasing one random vector access per
// candidate.
//
// Everything specific to that region lives here: its geometry (QgLayout) and
// its one-shot materialization (QgMaterialize).  The graph layer only stores
// the region and hands it to the quantizer capability that consumes it, so it
// needs no knowledge of the code layout itself.

#pragma once

#include <cstddef>
#include <cstdint>
// Rooted at src/ so this header stays includable from core (see ivf_entity).
#include <turbo/quantizer/common/pq_quantizer/packed_code_quantizer.h>

namespace zvec {
namespace core {

//! Geometry of the QG region inside a node record.
//!
//! The region is described in terms of the consuming quantizer's packed
//! layout: how many vectors one packed block covers and how many bytes it
//! takes.  Both come from the quantizer, so no quantization detail leaks into
//! the graph storage.  block_bytes == 0 disables the region.
struct QgLayout {
  //! Vectors covered by one packed block
  uint32_t block_vectors{0U};
  //! Byte size of one packed block
  uint32_t block_bytes{0U};
  //! Bytes reserved per node record (0 when disabled)
  uint32_t region_size{0U};
  //! Region offset inside the node record
  uint32_t offset{0U};

  //! Whether a region is reserved in every node record
  bool enabled() const {
    return region_size != 0U;
  }

  //! Packed blocks needed to cover `degree` neighbors
  size_t blocks_per_node(size_t degree) const {
    if (block_vectors == 0U) {
      return 0UL;
    }
    return (degree + block_vectors - 1UL) / block_vectors;
  }

  //! Bytes needed to cover `degree` neighbors
  size_t region_bytes(size_t degree) const {
    return blocks_per_node(degree) * block_bytes;
  }

  //! Place the region right after `base_size` bytes of node record (vector +
  //! key + level 0 neighbor ids, i.e. appended so that no existing offset
  //! moves), fill in offset / region_size, and return the resulting node size.
  //!
  //! The region start and the node size are both cache-line aligned so that
  //! every node's region begins on a 64-byte boundary for the wide SIMD loads
  //! that scan it.  When the region is disabled the node size keeps the plain
  //! 32-byte record alignment the graph uses otherwise.
  size_t configure(size_t base_size, size_t degree);
};

//! Narrow view of the graph storage used while materializing the QG region.
//!
//! Copy semantics on purpose: it keeps the materializer independent of the
//! node layout and safe for storage backends whose pointers are not stable.
class QgGraphView {
 public:
  virtual ~QgGraphView() = default;

  //! Node count; ids to materialize are [0, doc_count())
  virtual uint32_t doc_count() const = 0;

  //! Byte length of one stored code
  virtual size_t code_length() const = 0;

  //! Copy the level 0 neighbor ids of `id` into `out`, whose capacity is the
  //! maximum level 0 degree passed to QgMaterialize.  Returns the count.
  virtual size_t copy_l0_neighbors(uint32_t id, uint32_t *out) const = 0;

  //! Copy the code_length() code bytes of `id` into `out`
  virtual int copy_code(uint32_t id, void *out) const = 0;

  //! Write one packed block into the QG region of `id`
  virtual int write_block(uint32_t id, size_t block_index, const void *data,
                          size_t len) = 0;
};

//! Fill the QG region of every node with the packed codes of its level 0
//! neighbors.
//!
//! The caller must hold exclusive access while this runs, and must invalidate
//! the region on any later graph mutation (a rebuild then restores it).
//! Blocks not covered by a node's actual degree are zero filled, so a scan of
//! the whole region is always well defined.
//!
//! Cost is O(doc_count * max_degree): every call rebuilds the whole region,
//! there is no dirty-node tracking yet.
//!
//! @param max_degree maximum level 0 degree, sizing the neighbor buffers
//! @param packer the packed-code capability of the active quantizer
//! @param thread_count 0 selects the hardware concurrency
int QgMaterialize(QgGraphView &graph, const QgLayout &layout, size_t max_degree,
                  const zvec::turbo::PackedCodeQuantizer *packer,
                  uint32_t thread_count);

}  // namespace core
}  // namespace zvec
