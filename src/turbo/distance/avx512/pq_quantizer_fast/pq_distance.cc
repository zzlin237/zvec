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

// This file is compiled with the AVX-512 march flag (set in CMakeLists.txt
// for the whole distance/avx512 directory). _mm512_shuffle_epi8 additionally
// requires AVX512BW, hence the guard below: when the toolchain cannot emit
// the kernel, it forwards to the AVX2 one, which itself falls back to the
// scalar kernel when needed.

#include "avx512/pq_quantizer_fast/pq_distance.h"
#if defined(__AVX512BW__)
#include <immintrin.h>
#endif
#include <cstddef>
#include <cstdint>
#include "avx2/pq_quantizer_fast/pq_distance.h"
#include "common/fast_scan_common.h"

namespace zvec::turbo::avx512 {

#if defined(__AVX512BW__)
namespace {

// Sum the two 128-bit halves of one 256-bit uint16 accumulator into 8 int32
// lanes (same lane-pair merge the AVX2 kernel does in widen_lane_sum).
inline __m256i widen_lane_sum(__m256i s) {
  return _mm256_add_epi32(
      _mm256_cvtepu16_epi32(_mm256_castsi256_si128(s)),
      _mm256_cvtepu16_epi32(_mm256_extracti128_si256(s, 1)));
}

// Widen a uint16 partial sum holding FOUR 128-bit lanes (two sub-quantizer
// pairs) into one int32 accumulator: every group of four lanes contributes
// to the same 8 vectors, so they are all added together.
inline __m512i widen_quad_sum(__m512i s) {
  __m256i w_lo = widen_lane_sum(_mm512_extracti64x4_epi64(s, 0));
  __m256i w_hi = widen_lane_sum(_mm512_extracti64x4_epi64(s, 1));
  return _mm512_inserti64x4(_mm512_castsi256_si512(w_lo), w_hi, 1);
}

// Scan N back-to-back packed blocks (N = 1 or 2) with hoisted constants.
// Every LUT load serves all N blocks of the tile: the 32-zmm budget fits two
// full block states (16 zmm) plus masks / temporaries, so a 2-block tile
// halves the LUT traffic of the scan.
template <size_t N>
void fast_scan_blocks_impl(const uint8_t *packed_codes,
                           const uint8_t *packed_lut, size_t nsq_even,
                           __m512i low_mask, __m512i u16_mask,
                           int32_t *scores) {
  static_assert(N == 1 || N == 2, "tile size must be 1 or 2");
  // Pairs of sub-quantizers between two int32 spills. Each u16 slot gains at
  // most 2 * 255 per pair (two lanes), so 128 pairs stays well below 65535
  // whether pairs arrive one (32B step) or two (64B step) at a time.
  constexpr size_t kSpillPeriod = 128;
  const size_t block_bytes = nsq_even * 16;

  // int32 accumulators, one 64-byte register per contiguous group of 8
  // vectors; each 128-bit lane is the merge of one sub-quantizer's partials.
  __m512i acc_a[N], acc_b[N], acc_c[N], acc_d[N];
  // uint16 partial sums, four lanes (two sub-quantizer pairs) wide.
  __m512i s_a[N], s_b[N], s_c[N], s_d[N];
  for (size_t i = 0; i < N; ++i) {
    acc_a[i] = _mm512_setzero_si512();  // vectors  0..7
    acc_b[i] = _mm512_setzero_si512();  // vectors 16..23
    acc_c[i] = _mm512_setzero_si512();  // vectors  8..15
    acc_d[i] = _mm512_setzero_si512();  // vectors 24..31
    s_a[i] = _mm512_setzero_si512();
    s_b[i] = _mm512_setzero_si512();
    s_c[i] = _mm512_setzero_si512();
    s_d[i] = _mm512_setzero_si512();
  }

  size_t pending = 0;
  auto spill = [&]() {
    for (size_t i = 0; i < N; ++i) {
      acc_a[i] = _mm512_add_epi32(acc_a[i], widen_quad_sum(s_a[i]));
      acc_b[i] = _mm512_add_epi32(acc_b[i], widen_quad_sum(s_b[i]));
      acc_c[i] = _mm512_add_epi32(acc_c[i], widen_quad_sum(s_c[i]));
      acc_d[i] = _mm512_add_epi32(acc_d[i], widen_quad_sum(s_d[i]));
      s_a[i] = _mm512_setzero_si512();
      s_b[i] = _mm512_setzero_si512();
      s_c[i] = _mm512_setzero_si512();
      s_d[i] = _mm512_setzero_si512();
    }
    pending = 0;
  };

  size_t m = 0;
  // The packing contract (fast_scan_common.h) pads an odd num_chunk to an
  // even count only, so a single leading pair may remain once the 4-way
  // loop is done; process it with plain 32-byte steps like the AVX2 kernel.
  // Main loop: two sub-quantizer pairs (four lanes) per iteration.
  for (; m + 4 <= nsq_even; m += 4) {
    // Lane i = sub-quantizer m + i, for both the codes and the LUT:
    // _mm512_shuffle_epi8 looks up each 128-bit lane independently.
    const __m512i table = _mm512_loadu_si512(packed_lut + m * 16);
    for (size_t i = 0; i < N; ++i) {
      __m512i codes =
          _mm512_loadu_si512(packed_codes + i * block_bytes + m * 16);

      // No 8-bit shift: shift as uint16 then mask off the nibble that bled
      // in from the neighbouring byte.
      __m512i lo = _mm512_and_si512(codes, low_mask);
      __m512i hi = _mm512_and_si512(_mm512_srli_epi16(codes, 4), low_mask);

      // Byte j of r0 / r1 is the LUT entry of vector mapper[2j] / [2j+1].
      __m512i r0 = _mm512_shuffle_epi8(table, lo);
      __m512i r1 = _mm512_shuffle_epi8(table, hi);

      // Splitting even / odd bytes yields four contiguous vector groups (see
      // kFastScanMapper): r0 even -> 0..7, r0 odd -> 16..23,
      //                   r1 even -> 8..15, r1 odd -> 24..31.
      s_a[i] = _mm512_add_epi16(s_a[i], _mm512_and_si512(r0, u16_mask));
      s_b[i] = _mm512_add_epi16(s_b[i], _mm512_srli_epi16(r0, 8));
      s_c[i] = _mm512_add_epi16(s_c[i], _mm512_and_si512(r1, u16_mask));
      s_d[i] = _mm512_add_epi16(s_d[i], _mm512_srli_epi16(r1, 8));
    }
    if (++pending == kSpillPeriod) {
      spill();
    }
  }
  // Leading / trailing single pair: same pair-per-iteration body, 32B loads.
  for (; m < nsq_even; m += 2) {
    __m256i table = _mm256_loadu_si256(
        reinterpret_cast<const __m256i *>(packed_lut + m * 16));
    const __m256i low_mask256 = _mm256_set1_epi8(0x0F);
    const __m256i u16_mask256 = _mm256_set1_epi16(0x00FF);
    for (size_t i = 0; i < N; ++i) {
      __m256i codes = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(
          packed_codes + i * block_bytes + m * 16));
      __m256i lo = _mm256_and_si256(codes, low_mask256);
      __m256i hi = _mm256_and_si256(_mm256_srli_epi16(codes, 4), low_mask256);
      __m256i r0 = _mm256_shuffle_epi8(table, lo);
      __m256i r1 = _mm256_shuffle_epi8(table, hi);
      // Two of the four lanes stay zero; widen_quad_sum merges correctly.
      s_a[i] = _mm512_add_epi16(
          s_a[i], _mm512_castsi256_si512(_mm256_and_si256(r0, u16_mask256)));
      s_b[i] = _mm512_add_epi16(
          s_b[i], _mm512_castsi256_si512(_mm256_srli_epi16(r0, 8)));
      s_c[i] = _mm512_add_epi16(
          s_c[i], _mm512_castsi256_si512(_mm256_and_si256(r1, u16_mask256)));
      s_d[i] = _mm512_add_epi16(
          s_d[i], _mm512_castsi256_si512(_mm256_srli_epi16(r1, 8)));
    }
    if (++pending == kSpillPeriod) {
      spill();
    }
  }

  // Flush the trailing partials.
  spill();

  // Sum the four lanes of each accumulator: vector v appears in lane v / 8
  // exactly once, so lane 0 + lane 1 + lane 2 + lane 3 holds its total in
  // the lane holding its group. Store in group order a, c, b, d (baked into
  // kFastScanMapper), matching the AVX2 kernel.
  for (size_t i = 0; i < N; ++i) {
    int32_t *accu32 = scores + i * kFastScanBlockSize;
    __m256i out_a = _mm256_add_epi32(_mm512_extracti64x4_epi64(acc_a[i], 0),
                                     _mm512_extracti64x4_epi64(acc_a[i], 1));
    __m256i out_b = _mm256_add_epi32(_mm512_extracti64x4_epi64(acc_b[i], 0),
                                     _mm512_extracti64x4_epi64(acc_b[i], 1));
    __m256i out_c = _mm256_add_epi32(_mm512_extracti64x4_epi64(acc_c[i], 0),
                                     _mm512_extracti64x4_epi64(acc_c[i], 1));
    __m256i out_d = _mm256_add_epi32(_mm512_extracti64x4_epi64(acc_d[i], 0),
                                     _mm512_extracti64x4_epi64(acc_d[i], 1));
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(accu32), out_a);
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(accu32 + 8), out_c);
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(accu32 + 16), out_b);
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(accu32 + 24), out_d);
  }
}

}  // namespace
#endif

void pq_adc_fast_scan_avx512(const void *packed_codes_v,
                             const void *packed_lut_v, size_t num_chunk,
                             int32_t *accu32) {
#if defined(__AVX512BW__)
  const auto *packed_codes = reinterpret_cast<const uint8_t *>(packed_codes_v);
  const auto *packed_lut = reinterpret_cast<const uint8_t *>(packed_lut_v);
  const size_t nsq_even = fast_scan_even_chunk(num_chunk);
  const __m512i low_mask = _mm512_set1_epi8(0x0F);
  const __m512i u16_mask = _mm512_set1_epi16(0x00FF);
  fast_scan_blocks_impl<1>(packed_codes, packed_lut, nsq_even, low_mask,
                           u16_mask, accu32);
#else
  // Unlike the float-returning PQ kernels, a no-op stub here would leave
  // accu32 untouched and silently yield zero distances, so forward instead.
  avx2::pq_adc_fast_scan_avx2(packed_codes_v, packed_lut_v, num_chunk, accu32);
#endif
}

void pq_adc_fast_scan_multi_avx512(const void *packed_codes_v,
                                   const void *packed_lut_v, size_t num_chunk,
                                   size_t num_blocks, int32_t threshold,
                                   int32_t *scores, uint32_t *masks) {
#if defined(__AVX512BW__)
  const size_t block_bytes = fast_scan_packed_block_size(num_chunk);
  const size_t nsq_even = fast_scan_even_chunk(num_chunk);
  const auto *packed_codes = reinterpret_cast<const uint8_t *>(packed_codes_v);
  const auto *packed_lut = reinterpret_cast<const uint8_t *>(packed_lut_v);
  const __m512i low_mask = _mm512_set1_epi8(0x0F);
  const __m512i u16_mask = _mm512_set1_epi16(0x00FF);

  // 2-block tiles: each LUT load serves both blocks of the tile (the 32-zmm
  // budget fits two full block states plus masks / temporaries).
  size_t b = 0;
  for (; b + 2 <= num_blocks; b += 2) {
    fast_scan_blocks_impl<2>(packed_codes + b * block_bytes, packed_lut,
                             nsq_even, low_mask, u16_mask,
                             scores + b * kFastScanBlockSize);
  }
  if (b < num_blocks) {
    fast_scan_blocks_impl<1>(packed_codes + b * block_bytes, packed_lut,
                             nsq_even, low_mask, u16_mask,
                             scores + b * kFastScanBlockSize);
  }

  for (size_t k = 0; k < num_blocks; ++k) {
    uint32_t mask = 0;
    const int32_t *block_scores = scores + k * kFastScanBlockSize;
    for (size_t j = 0; j < kFastScanBlockSize; ++j) {
      if (block_scores[j] < threshold) {
        mask |= 1u << j;
      }
    }
    masks[k] = mask;
  }
#else
  avx2::pq_adc_fast_scan_multi_avx2(packed_codes_v, packed_lut_v, num_chunk,
                                    num_blocks, threshold, scores, masks);
#endif
}

}  // namespace zvec::turbo::avx512
