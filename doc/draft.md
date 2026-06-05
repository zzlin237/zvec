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

### 1: 新增参数定义
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
  2. 保存矩阵（通过IndexDumper写入segment，含CRC + 32字节对齐）
  3. 加载矩阵（通过IndexStorage读取segment，含CRC校验）
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

  //! Return the serialized size of the rotator in bytes
  size_t dump_bytes() const;

  //! Dump the rotator data to an IndexDumper as a named segment.
  //! Writes the raw rotator bytes, appends padding for 32-byte alignment,
  //! and registers the segment meta (id, size, padding, crc).
  int dump(const IndexDumper::Pointer &dumper,
           const std::string &seg_id = RECORD_ROTATOR_SEG_ID) const;

  //! Load the rotator data from an IndexStorage segment.
  //! Reads the serialized rotator bytes and reconstructs the rotator.
  int load(IndexStorage::Pointer storage,
           const std::string &seg_id = RECORD_ROTATOR_SEG_ID);

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
