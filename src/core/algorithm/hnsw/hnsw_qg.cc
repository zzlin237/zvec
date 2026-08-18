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

#include "hnsw_qg.h"
#include <algorithm>
#include <atomic>
#include <memory>
#include <thread>
#include <vector>
#include <zvec/ailego/logger/logger.h>
#include <zvec/core/framework/index_error.h>
#include <zvec/core/framework/index_threads.h>

namespace zvec {
namespace core {

namespace {

//! Node record alignment used when no QG region is present, matching the
//! graph's own record alignment.
inline size_t AlignRecord(size_t size) {
  return (size + 0x1FUL) & (~0x1FUL);
}

//! Cache-line alignment for the QG region and for records carrying one.
inline size_t AlignCacheLine(size_t size) {
  return (size + 0x3FUL) & (~0x3FUL);
}

}  // namespace

size_t QgLayout::configure(size_t base_size, size_t degree) {
  const size_t bytes = region_bytes(degree);
  if (bytes == 0UL) {
    offset = 0U;
    region_size = 0U;
    return AlignRecord(base_size);
  }

  const size_t start = AlignCacheLine(base_size);
  offset = static_cast<uint32_t>(start);
  region_size = static_cast<uint32_t>(bytes);
  return AlignCacheLine(start + bytes);
}

int QgMaterialize(QgGraphView &graph, const QgLayout &layout, size_t max_degree,
                  const zvec::turbo::PackedCodeQuantizer *packer,
                  uint32_t thread_count) {
  if (packer == nullptr) {
    LOG_ERROR("Materialize quantized graph without a packer");
    return IndexError_InvalidArgument;
  }
  if (!layout.enabled() || layout.block_vectors == 0U) {
    LOG_ERROR("Materialize quantized graph with no region reserved");
    return IndexError_Unsupported;
  }

  const uint32_t docs = graph.doc_count();
  if (docs == 0U) {
    return 0;
  }

  const size_t code_len = graph.code_length();
  const size_t block_vectors = layout.block_vectors;
  const size_t block_bytes = layout.block_bytes;
  const size_t blocks = layout.blocks_per_node(max_degree);

  if (thread_count == 0U) {
    thread_count = std::thread::hardware_concurrency();
    if (thread_count == 0U) {
      thread_count = 1U;
    }
  }
  //! One task per thread over a contiguous id range: neighbouring nodes share
  //! storage chunks, so contiguous ranges keep the gather reads local.
  const uint32_t tasks = std::min<uint32_t>(thread_count, docs);
  const uint32_t per_task = (docs + tasks - 1U) / tasks;

  auto threads = std::make_shared<SingleQueueIndexThreads>(tasks, false);
  auto task_group = threads->make_group();
  std::atomic<int> failure{0};

  for (uint32_t t = 0; t < tasks; ++t) {
    const uint32_t begin = t * per_task;
    const uint32_t end = std::min<uint32_t>(begin + per_task, docs);
    if (begin >= end) {
      continue;
    }
    task_group->submit(ailego::Closure::New(
        [&graph, &layout, &failure, begin, end, code_len, block_vectors,
         block_bytes, blocks, max_degree, packer]() {
          //! Per task scratch: the gathered codes of one block laid out
          //! contiguously (what the packer expects) plus the packed block
          //! itself.
          std::vector<uint8_t> codes(block_vectors * code_len, 0);
          std::vector<uint8_t> block(block_bytes, 0);
          std::vector<uint32_t> neighbors(max_degree, 0U);

          for (uint32_t id = begin; id < end; ++id) {
            if (failure.load(std::memory_order_relaxed) != 0) {
              return;
            }
            const size_t count = graph.copy_l0_neighbors(id, neighbors.data());

            for (size_t b = 0; b < blocks; ++b) {
              const size_t first = b * block_vectors;
              const size_t num =
                  first < count ? std::min(block_vectors, count - first) : 0UL;
              for (size_t i = 0; i < num; ++i) {
                int ret = graph.copy_code(neighbors[first + i],
                                          codes.data() + i * code_len);
                if (ret != 0) {
                  LOG_ERROR("Failed to read code of neighbor %u of node %u",
                            neighbors[first + i], id);
                  failure.store(ret, std::memory_order_relaxed);
                  return;
                }
              }
              //! Slots beyond `num` are zero filled by the packer, so a scan of
              //! a partially filled block stays well defined.
              int ret =
                  packer->pack_codes(codes.data(), num, code_len, block.data());
              if (ret != 0) {
                LOG_ERROR("Failed to pack quantized graph block %zu of node %u",
                          b, id);
                failure.store(ret, std::memory_order_relaxed);
                return;
              }
              ret = graph.write_block(id, b, block.data(), block_bytes);
              if (ret != 0) {
                LOG_ERROR(
                    "Failed to write quantized graph block %zu of node %u", b,
                    id);
                failure.store(ret, std::memory_order_relaxed);
                return;
              }
            }
          }
        }));
  }
  task_group->wait_finish();

  int ret = failure.load(std::memory_order_relaxed);
  if (ret != 0) {
    return ret;
  }

  LOG_INFO(
      "Materialized quantized graph, docs=%u blocks/node=%zu region=%u bytes",
      docs, blocks, layout.region_size);
  return 0;
}

}  // namespace core
}  // namespace zvec
