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

#include <cstddef>
#include <cstdint>

namespace zvec {
namespace turbo {

//! Optional capability: a quantizer whose codes must be stored in packed
//! 32-vector blocks (e.g. FastScan, where one SIMD byte shuffle looks up
//! 32 codes per sub-space).  Orthogonal to the Quantizer base contract,
//! so storage layers discover it via dynamic_cast: a successful cast both
//! requires packing and provides the packer.  Storage layers that cannot
//! honor packing must not use such a quantizer.
//!
//! The write side (pack_codes) and the read side (calc_distance_packed_block)
//! share the same block layout contract; generic gather-style quantizers
//! keep using Quantizer::calc_distance_dp_query_batch.
class PackedCodeQuantizer {
 public:
  virtual ~PackedCodeQuantizer() = default;

  //! Pack up to 32 plain codes (laid out `stride` bytes apart) into one
  //! packed block consumable by calc_distance_packed_block.  Slots beyond
  //! `num` are zero-filled; `out` must hold one full packed block.
  virtual int pack_codes(const void *codes, size_t num, size_t stride,
                         void *out) const = 0;

  //! Scan one or several back-to-back packed blocks (produced by
  //! pack_codes) against a quantized query, writing `num` distances.
  virtual void calc_distance_packed_block(const void *block, size_t num,
                                          const void *query,
                                          float *dist_list) const = 0;

  //! Fused multi-block scan: compares an integer score threshold inside the
  //! scan (INT32_MAX = no pruning) and dequantizes surviving vectors only.
  //! Writes survivor distances and their batch-global slots (indexing the
  //! contiguous `num`-vector input), and
  //! returns the survivor count.  `num` is the actual vector count; padded
  //! slots of a trailing block never appear in the output.  The threshold is
  //! expressed in the quantizer's own integer score domain (see
  //! packed_score_threshold).
  virtual size_t scan_packed_blocks(const void *blocks, size_t num,
                                    const void *query, int32_t threshold,
                                    float *survivor_dists,
                                    uint32_t *survivor_slots) const = 0;

  //! Invert a heap-top threshold into the integer score domain.  Heap values
  //! live in the affine domain (dist + add) * mul, where add / mul are
  //! generic affine coefficients supplied by the caller (IVF passes its
  //! residual_base / norm_val); the quantizer inverts them together with its
  //! own delta / bias without knowing their semantics.  Returns INT32_MAX
  //! when pruning is unavailable (non-positive mul, non-finite heap top,
  //! degenerate delta).
  virtual int32_t packed_score_threshold(const void *query, float heap_top,
                                         float add, float mul) const = 0;
};

}  // namespace turbo
}  // namespace zvec
