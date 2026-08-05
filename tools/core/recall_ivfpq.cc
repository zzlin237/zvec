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

#include "helper.h"

// -------------------------------------------------------
// RecallIVFPQ: recall for IVF + per-cluster residual PQ with optional refiner
// -------------------------------------------------------
// This tool evaluates recall for an IVF index whose posting lists store
// per-cluster residual PQ codes (pure PQ-ADC search), and optionally tests the
// refiner path that re-ranks the PQ coarse candidates using a flat FP32
// reference index.
//
// The refiner over-fetches (scale_factor * topk) candidates from the IVF+PQ
// index, then recomputes exact FP32 distances over those candidate ids via the
// reference flat index (brute force by primary key) and keeps the final topk.
//
// Usage: recall_ivfpq CONFIG.yaml [plugin files...]
//
// The YAML config must contain:
//   IndexCommon:
//     IndexConfig, IndexPath, TopK, QueryFile, QueryType, ...
//   QueryConfig:
//     QueryParam: '{"index_type":"kIVF","nprobe":64}'
//     RefinerConfig:          (optional, enables refiner test)
//       ScaleFactor: !!int 4  (coarse candidates = ScaleFactor * topk)
//       ReferenceIndex:
//         Config: '{"use_id_map":false,...,"index_type":"kFlat",...}'
//         Path: /path/to/flat.index
// -------------------------------------------------------

mutex g_recall_lock;

static bool load_queries(const string &query_file, const string &first_sep,
                         const string &second_sep,
                         vector<vector<float>> &queries, size_t &dim) {
  TxtInputReader<float> reader;
  vector<SparseData<float>> sparse_data;
  vector<vector<uint64_t>> taglists;
  if (!reader.load_query(query_file, first_sep, second_sep, queries,
                         sparse_data, taglists)) {
    LOG_ERROR("Failed to load query file");
    return false;
  }
  if (queries.empty()) {
    LOG_ERROR("No queries loaded");
    return false;
  }
  dim = queries[0].size();
  cout << "Loaded " << queries.size() << " queries, dim=" << dim << endl;
  return true;
}

// Generate ground truth by linear search on the reference (flat) index
static bool generate_ground_truth(
    core_interface::Index::Pointer flat_index,
    const vector<vector<float>> &queries, size_t gt_count,
    vector<vector<pair<uint64_t, float>>> &gt, shared_ptr<ThreadPool> pool) {
  cout << "Generating ground truth from flat index (gt_count=" << gt_count
       << ")..." << endl;

  gt.resize(queries.size());
  atomic_bool error(false);
  atomic_size_t count(0);
  size_t total = queries.size();

  function<void(size_t)> fun = [&](size_t i) {
    auto c = ++count;
    if (c % 100 == 0 || c == total) {
      cout << "\rGT progress: " << c << "/" << total << flush;
    }

    auto query_param = std::make_shared<core_interface::FlatQueryParam>();
    query_param->topk = gt_count;
    query_param->is_linear = true;

    core_interface::DenseVector dense_query;
    dense_query.data = const_cast<float *>(queries[i].data());
    core_interface::VectorData query_data;
    query_data.vector = dense_query;

    core_interface::SearchResult search_result;
    int ret = flat_index->Search(query_data, query_param, &search_result);
    if (ret < 0) {
      LOG_ERROR("GT linear search failed, ret=%d, idx=%zu", ret, i);
      error.exchange(true);
      return;
    }

    vector<pair<uint64_t, float>> one_gt;
    one_gt.reserve(gt_count);
    for (auto &knn : search_result.doc_list_) {
      one_gt.emplace_back(knn.key(), knn.score());
    }
    gt[i] = std::move(one_gt);
  };

  for (size_t i = 0; i < queries.size(); ++i) {
    if (error) break;
    pool->enqueue_and_wake(Closure::New(fun, i));
  }
  pool->wait_finish();
  cout << endl;

  if (error) {
    LOG_ERROR("Ground truth generation failed");
    return false;
  }
  cout << "Ground truth generated for " << gt.size() << " queries" << endl;
  return true;
}

// Compute recall@K by comparing keys (ID-based comparison)
static void compute_recall_by_id(
    const vector<IndexDocument> &knn_res,
    const vector<pair<uint64_t, float>> &one_gt, size_t topk,
    map<size_t, float> &recall_acc) {
  size_t result_size = std::min(topk, one_gt.size());
  if (result_size == 0) return;

  for (size_t k = 1; k <= result_size; ++k) {
    size_t match = 0;
    set<uint64_t> topk_keys;
    for (size_t i = 0; i < knn_res.size() && i < k; ++i) {
      topk_keys.insert(knn_res[i].key());
    }
    for (size_t i = 0; i < k && i < one_gt.size(); ++i) {
      if (topk_keys.count(one_gt[i].first)) {
        match++;
      }
    }
    auto it = recall_acc.find(k);
    if (it != recall_acc.end()) {
      lock_guard<mutex> lock(g_recall_lock);
      it->second += 100.0 * match / k;
    }
  }
}

// Run recall test for a single configuration
static void run_recall_test(
    core_interface::Index::Pointer pq_index,
    core_interface::BaseIndexQueryParam::Pointer query_param,
    const vector<vector<float>> &queries, size_t topk,
    const vector<vector<pair<uint64_t, float>>> &gt,
    const vector<int32_t> &topk_ids, const string &label,
    shared_ptr<ThreadPool> pool) {
  cout << "\n======================================" << endl;
  cout << "  Recall test: " << label << endl;
  cout << "======================================" << endl;

  map<size_t, float> recall_res;
  for (auto k : topk_ids) {
    recall_res[k] = 0.0f;
  }

  atomic_size_t count(0);
  size_t total = queries.size();

  function<void(size_t)> fun = [&](size_t i) {
    auto c = ++count;
    if (c % 100 == 0 || c == total) {
      cout << "\rSearch progress: " << c << "/" << total << flush;
    }

    core_interface::DenseVector dense_query;
    dense_query.data = const_cast<float *>(queries[i].data());
    core_interface::VectorData query_data;
    query_data.vector = dense_query;

    auto query_param_clone = query_param->Clone();
    query_param_clone->topk = topk;
    query_param_clone->is_linear = false;

    core_interface::SearchResult search_result;
    int ret = pq_index->Search(query_data, query_param_clone, &search_result);
    if (ret < 0) {
      LOG_ERROR("Search failed, ret=%d, idx=%zu", ret, i);
      return;
    }

    compute_recall_by_id(search_result.doc_list_, gt[i], topk, recall_res);
  };

  for (size_t i = 0; i < queries.size(); ++i) {
    pool->enqueue_and_wake(Closure::New(fun, i));
  }
  pool->wait_finish();
  cout << endl;

  cout << "  Results for [" << label << "]:" << endl;
  for (auto &it : recall_res) {
    cout << "    Recall@" << it.first << ": " << it.second / queries.size()
         << "%" << endl;
  }
}

static bool check_config(YAML::Node &config_node) {
  auto common = config_node["IndexCommon"];
  if (!common) {
    LOG_ERROR("Can not find [IndexCommon] in config");
    return false;
  }
  if (!common["IndexConfig"]) {
    LOG_ERROR("Can not find [IndexConfig] in config");
    return false;
  }
  if (!common["IndexPath"]) {
    LOG_ERROR("Can not find [IndexPath] in config");
    return false;
  }
  if (!common["TopK"]) {
    LOG_ERROR("Can not find [TopK] in config");
    return false;
  }
  if (!common["QueryFile"]) {
    LOG_ERROR("Can not find [QueryFile] in config");
    return false;
  }
  auto query_config = config_node["QueryConfig"];
  if (!query_config) {
    LOG_ERROR("Can not find [QueryConfig] in config");
    return false;
  }
  if (!query_config["QueryParam"]) {
    LOG_ERROR("Can not find [QueryConfig.QueryParam] in config");
    return false;
  }
  return true;
}

static void usage(void) {
  cout << "Usage: recall_ivfpq CONFIG.yaml [plugin file path]" << endl;
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    usage();
    return -1;
  }

  // Load plugins
  IndexPluginBroker broker;
  std::string error;
  for (int i = 2; i < argc; ++i) {
    if (!broker.emplace(argv[i], &error)) {
      LOG_ERROR("Failed to load plugin: %s (%s)", argv[i], error.c_str());
      return -1;
    }
  }

  // Load config
  YAML::Node config_node;
  try {
    config_node = YAML::LoadFile(argv[1]);
  } catch (...) {
    LOG_ERROR("Load YAML file[%s] failed!", argv[1]);
    return -1;
  }
  if (!check_config(config_node)) {
    return -1;
  }

  auto config_common = config_node["IndexCommon"];

  // Log level
  map<string, int> LOG_LEVEL_MAP = {{"debug", IndexLogger::LEVEL_DEBUG},
                                    {"info", IndexLogger::LEVEL_INFO},
                                    {"warn", IndexLogger::LEVEL_WARN},
                                    {"error", IndexLogger::LEVEL_ERROR},
                                    {"fatal", IndexLogger::LEVEL_FATAL}};
  string log_level = config_common["LogLevel"]
                         ? config_common["LogLevel"].as<string>()
                         : "info";
  transform(log_level.begin(), log_level.end(), log_level.begin(), ::tolower);
  if (LOG_LEVEL_MAP.find(log_level) != LOG_LEVEL_MAP.end()) {
    IndexLoggerBroker::SetLevel(LOG_LEVEL_MAP[log_level]);
    zvec::ailego::LoggerBroker::SetLevel(LOG_LEVEL_MAP[log_level]);
  }

  // Thread pool
  size_t thread_count = config_common["RecallThreadCount"]
                            ? config_common["RecallThreadCount"].as<uint64_t>()
                            : 0;
  shared_ptr<ThreadPool> pool;
  if (thread_count == 0) {
    pool = make_shared<ThreadPool>();
    thread_count = pool->count();
    cout << "Using CPU count as thread pool count[" << thread_count << "]"
         << endl;
  } else {
    pool = make_shared<ThreadPool>(thread_count, false);
    cout << "Using thread pool count[" << thread_count << "]" << endl;
  }

  // GT count
  size_t gt_count = config_common["RecallGTCount"]
                        ? config_common["RecallGTCount"].as<uint64_t>()
                        : 100;

  // TopK
  string top_k_str = config_common["TopK"].as<string>();
  vector<int32_t> topk_ids;
  StringHelper::Split(top_k_str, ",", &topk_ids);
  std::sort(topk_ids.begin(), topk_ids.end());
  size_t topk = topk_ids.back();
  gt_count = topk < gt_count ? gt_count : topk;

  // Query config
  string query_file = config_common["QueryFile"].as<string>();
  string first_sep = config_common["QueryFirstSep"]
                         ? config_common["QueryFirstSep"].as<string>()
                         : ";";
  string second_sep = config_common["QuerySecondSep"]
                          ? config_common["QuerySecondSep"].as<string>()
                          : " ";

  // Load queries
  vector<vector<float>> queries;
  size_t dim = 0;
  if (!load_queries(query_file, first_sep, second_sep, queries, dim)) {
    return -1;
  }

  // Load IVF+PQ index (main index under test)
  string index_dir = config_common["IndexPath"].as<string>();
  core_interface::Index::Pointer pq_index;
  core_interface::BaseIndexQueryParam::Pointer query_param;
  if (parse_and_load_index_param(config_node, index_dir, pq_index,
                                 query_param) != 0) {
    LOG_ERROR("Failed to load IVF+PQ index");
    return -1;
  }

  // Load flat reference index (used for GT generation and for the refiner).
  core_interface::Index::Pointer flat_index = nullptr;
  bool has_refiner = false;
  int scale_factor = 4;

  auto query_config = config_node["QueryConfig"];
  if (auto refiner_config = query_config["RefinerConfig"]; refiner_config) {
    has_refiner = true;
    if (refiner_config["ScaleFactor"]) {
      scale_factor = refiner_config["ScaleFactor"].as<int>();
      if (scale_factor < 1) scale_factor = 1;
    }

    auto ref_index_config = refiner_config["ReferenceIndex"];
    if (ref_index_config && ref_index_config["Config"]) {
      auto params = core_interface::IndexFactory::DeserializeIndexParamFromJson(
          ref_index_config["Config"].as<std::string>());
      flat_index = core_interface::IndexFactory::CreateAndInitIndex(*params);

      if (ref_index_config["Path"]) {
        auto ref_path = ref_index_config["Path"].as<std::string>();
        core_interface::StorageOptions storage_options;
        storage_options.type =
            core_interface::StorageOptions::StorageType::kMMAP;
        storage_options.create_new = false;
        storage_options.read_only = true;

        int ret = flat_index->Open(ref_path, storage_options);
        if (ret != 0) {
          LOG_ERROR("Failed to open reference index at %s, ret=%d",
                    ref_path.c_str(), ret);
          return -1;
        }
        cout << "Loaded reference (flat) index from: " << ref_path << endl;
      } else {
        LOG_ERROR("RefinerConfig.ReferenceIndex.Path is required");
        return -1;
      }
    } else {
      LOG_ERROR("RefinerConfig.ReferenceIndex.Config is required");
      return -1;
    }
  } else if (config_common["FlatIndexPath"]) {
    // No refiner config - use a separate FlatIndexPath just for GT generation.
    string flat_path = config_common["FlatIndexPath"].as<string>();
    string flat_config_json = config_common["FlatIndexConfig"]
                                  ? config_common["FlatIndexConfig"].as<string>()
                                  : "";
    if (flat_config_json.empty()) {
      LOG_ERROR("FlatIndexConfig is required when FlatIndexPath is set");
      return -1;
    }
    auto params =
        core_interface::IndexFactory::DeserializeIndexParamFromJson(
            flat_config_json);
    flat_index = core_interface::IndexFactory::CreateAndInitIndex(*params);

    core_interface::StorageOptions storage_options;
    storage_options.type = core_interface::StorageOptions::StorageType::kMMAP;
    storage_options.create_new = false;
    storage_options.read_only = true;

    int ret = flat_index->Open(flat_path, storage_options);
    if (ret != 0) {
      LOG_ERROR("Failed to open flat index at %s, ret=%d", flat_path.c_str(),
                ret);
      return -1;
    }
    cout << "Loaded flat index for GT from: " << flat_path << endl;
  } else {
    LOG_ERROR("Either RefinerConfig.ReferenceIndex or FlatIndexPath must "
              "be provided for ground truth generation");
    return -1;
  }

  // Generate ground truth from flat index
  vector<vector<pair<uint64_t, float>>> gt;
  if (!generate_ground_truth(flat_index, queries, gt_count, gt, pool)) {
    LOG_ERROR("Failed to generate ground truth");
    return -1;
  }

  // ---- Test 1: IVF+PQ only (pure PQ-ADC, no refiner) ----
  {
    auto pq_query_param = query_param->Clone();
    pq_query_param->refiner_param = nullptr;
    run_recall_test(pq_index, pq_query_param, queries, topk, gt, topk_ids,
                    "IVF+PQ (no refiner)", pool);
  }

  // ---- Test 2: IVF+PQ + Refiner (over-fetch scale_factor*topk, re-rank FP32) --
  if (has_refiner && flat_index) {
    auto refiner_query_param = query_param->Clone();
    auto refiner_param = std::make_shared<core_interface::RefinerParam>();
    refiner_param->reference_index = flat_index;
    refiner_param->scale_factor_ = static_cast<float>(scale_factor);
    refiner_query_param->refiner_param = refiner_param;

    run_recall_test(pq_index, refiner_query_param, queries, topk, gt, topk_ids,
                    "IVF+PQ+Refiner (x" + std::to_string(scale_factor) + ")",
                    pool);
  }

  // Cleanup
  pq_index->Close();
  if (flat_index) {
    flat_index->Close();
  }

  cout << "\nRecall IVF+PQ test done." << endl;
  return 0;
}
