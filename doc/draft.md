## 量化方案新增旋转功能

1. 动机：
Int8量化采用 per-vector min-max 量化，即用每个向量自身的最小/最大值来确定量化区间 [-127, 127]，误差主要来自：
    - 维度间的值分布不均匀：某些维度的值远大于其他维度，导致量化区间被少数极端维度"撑开"，大部分维度的量化精度被浪费。
    - 非各向同性分布：真实embedding数据的能量往往集中在少数方向上。
随机旋转在保持距离不变的同时，会将向量的能量均匀分散到所有维度，使每个维度的值分布更接近高斯分布，从而减小per-vector min-max量化的量化误差。

2. 修改类型：
一种可选的量化参数
```yaml
// 构建侧新增量化配置选项：
ConverterParams:
    integer_streaming.converter.enable_rotate: !!bool true
// 搜索侧不做变化
```
```
Build 阶段:
  Converter::init()      → 读取 enable_rotate=true，创建 rabitqlib::Rotator
  Converter::transform() → 每条向量: rotator->rotate(x) → [normalize] → int8 量化
  Converter::dump()      → 将 rotator 数据写入独立 segment
  Streamer::dump()       → 写入 meta + HNSW 图数据（不感知 converter）
  meta.set_reformer()    → reformer_params 中写入 enable_rotate=true

Search 阶段:
  Index::Open()          → reformer_->load(storage_) → 从 segment 加载 rotator
  Reformer::transform()  → 每条 query: rotator->rotate(q) → [normalize] → int8 量化
```
## Int8StreamingConverter具体实现

### 1: 新增参数定义 [DONE]
```cpp
//! IntegerStreamingConverter
static const std::string INTEGER_STREAMING_CONVERTER_ENABLE_ROTATE =
    "integer_streaming.converter.enable_rotate";
//! IntegerStreamingReformer
static const std::string INTEGER_STREAMING_REFORMER_ENABLE_ROTATE =
    "integer_streaming.reformer.enable_rotate";
```

### 2. 新增矩阵旋转工具类 [DONE]
1. 便于拓展，将旋转功能抽象到统一的文件`/root/code/zvec/src/core/quantizer/record_rotater.h`和`record_rotater.cc`中（pimpl模式，rabitqlib依赖仅在.cc中）
2. 实现方式参考/root/code/zvec/src/core/algorithm/hnsw_rabitq中的旋转方式，具体实现调用第三方库/root/code/zvec/thirdparty/RaBitQ-Library
3. 包含功能：
  1. O(d \log d)复杂度的快速旋转
  2. dump：保存矩阵（通过IndexDumper写入segment，含自描述Header + rabitqlib blob + CRC + 32字节对齐）
  3. open：从Storage加载序列化旋转器（通过IndexStorage读取segment，从Header解析type/dim/padded_dim，无需预先init，含CRC校验）
  4. load：加载用户自定义旋转矩阵（MatrixRotator，行主序 dim x padded_dim）
```cpp
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

  //! Dump the rotator to an IndexDumper as a named segment.
  //! Format: [Header: type(1B)|origin_dim(4B)|padded_dim(4B)] [rabitqlib blob]
  //! Appends padding for 32-byte alignment, registers segment meta (id, size, padding, crc).
  int dump(const IndexDumper::Pointer &dumper,
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
```
### 3. 修改 IntegerStreaming 的 Converter 和 Reformer [DONE]

1. 修改文件：`integer_quantizer_converter.cc` 和 `integer_quantizer_reformer.cc`
2. Converter 修改：
  1. 新增 `#include "record_rotater.h"` 和成员变量 `enable_rotate_`, `rotator_`（无 `padded_dim_`，由 `rotator_->padded_dim()` 派生）
  2. `init()` 读取 `enable_rotate` 标记，创建 FhtKacRotator（padded_dim=向上取64倍数），将 `enable_rotate` 写入 reformer_params
  3. `transform()` 将 `rotator_` 传入 Holder，Holder 通过 `rotator_->padded_dim()` 获取对齐维度
  4. `dump()` 调用 `rotator_->dump(dumper)` 保存旋转矩阵（自描述格式）
  5. Holder Iterator 的 `encode_record()` 管线：rotate → normalize → quantize
3. Reformer 修改：
  1. `init()` 仅读取 `enable_rotate` 标记（维度信息从序列化数据自描述获取）
  2. `load()` 创建 rotator，调用 `rotator_->open(storage)` 加载旋转矩阵（open 内部从 header 解析 type/dim/padded_dim）
  3. 所有 `transform()`/`convert()` 方法在量化前应用旋转
  4. `revert()` 在旋转模式下拒绝反量化

### 4. 修改 Index::Open() [DONE]
1. 修改代码：`src/core/interface/index.cc`
2. 在 `Index::Open()` 中 streamer 打开后，调用 `reformer_->load(storage_)` 加载序列化数据（旋转矩阵等）
3. 对无序列化数据的 reformer（如非旋转模式），`load()` 为 no-op 直接返回 0，不干扰运行时功能

### 5. 修改运行时测试代码
0. 测试原始功能是否有问题：
```
./build/bin/bench /root/code/zvec/config/search_baseline.yaml
./build/bin/recall /root/code/zvec/config/search_baseline.yaml
```
查看是否能正常运行，以检查原始功能是否出现问题
1. 修改代码：/root/code/zvec/tools/core/local_builder.cc，使其可以保存旋转矩阵
2. 编译代码：
```cpp
cmake -DENABLE_SKYLAKE=ON -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```
3. 测试代码：
索引构建：
```cpp
./build/bin/local_builder /root/code/zvec/config/construct.yaml
```
搜索测试：
```cpp
./build/bin/bench /root/code/zvec/config/search_baseline.yaml
./build/bin/bench /root/code/zvec/config/search_current.yaml
```
4. 运行代码，并修改错误
