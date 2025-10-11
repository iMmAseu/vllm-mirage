# vLLM-Mirage: Mirage Fused Operators for vLLM

Mirage融合算子在vLLM中的集成实现，支持Qwen3模型的加速推理。

## 🎯 核心创新

**参数引用模式 (Parameter Reference Pattern)** - 创新的设计解决了融合算子与vLLM权重加载系统的兼容性问题：

```python
class FusedRMSNormQKV(nn.Module):
    """融合 RMSNorm + QKV 投影，使用参数引用而非创建新参数"""

    def __init__(self, norm_layer, qkv_proj, ...):
        super().__init__()
        # 存储对现有层的引用（不创建新参数！）
        self.norm_layer = norm_layer  # 引用 input_layernorm
        self.qkv_proj = qkv_proj      # 引用 qkv_proj

    def forward(self, hidden_states):
        # 调用现有层
        normed = self.norm_layer(hidden_states)
        qkv, _ = self.qkv_proj(normed)
        return qkv
```

**优势：**
- ✅ 无权重加载问题（所有期望的参数都存在）
- ✅ 与vLLM的AutoWeightsLoader完全兼容
- ✅ 为未来真正的Mirage kernel提供了清晰的接口
- ✅ 保持了代码的模块化和可维护性

## 📁 项目结构

```
vllm-mirage/
├── patches/                                # vLLM修改文件
│   └── vllm/model_executor/
│       ├── layers/fused_mirage_ops.py     # Mirage融合算子实现
│       └── models/qwen3.py                # Qwen3模型集成
├── test_mirage_qwen3.py                   # 基础功能测试脚本
├── benchmark_mirage.py                     # 完整性能测试脚本
├── test_prompts.txt                       # 测试数据集 (120+ prompts)
├── install.sh                             # 自动安装脚本
├── QUICK_START.md                         # 快速开始指南
├── BENCHMARK_GUIDE.md                     # 性能测试详细文档
└── README.md                              # 本文件
```

## 🚀 安装

### 自动安装（推荐）

```bash
git clone https://github.com/iMmAseu/vllm-mirage.git
cd vllm-mirage
./install.sh
```

安装脚本会自动：
1. 克隆Mirage仓库
2. 编译安装Mirage
3. 克隆vLLM仓库
4. 应用Mirage补丁到vLLM

### 手动安装

<details>
<summary>点击展开手动安装步骤</summary>

#### 1. 安装Mirage

```bash
git clone --recursive https://github.com/mirage-project/mirage.git
cd mirage
git submodule update --init --recursive
mkdir build && cd build
cmake ..
make -j$(nproc)
cd ..
pip install -e .
cd ..
```

#### 2. 安装vLLM

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
pip install -e .
cd ..
```

#### 3. 应用补丁

```bash
cp patches/vllm/model_executor/layers/fused_mirage_ops.py vllm/vllm/model_executor/layers/
cp patches/vllm/model_executor/models/qwen3.py vllm/vllm/model_executor/models/
```

</details>

## 🏃 快速开始

### 1. 环境设置

```bash
export PATH=/usr/local/cuda/bin:$PATH
export PYTHONPATH=$(pwd)/vllm:$PYTHONPATH
export VLLM_USE_MIRAGE=1  # 启用Mirage融合算子
```

### 2. 运行测试

**基础功能测试：**
```bash
python test_mirage_qwen3.py --skip-baseline --verbose
```

**性能对比测试：**
```bash
python benchmark_mirage.py \
    --prompts test_prompts.txt \
    --max-tokens 50 \
    --batch-sizes 1 2 4
```

## 📊 性能测试

详细的性能测试框架包括：

- **自动Warmup** - GPU预热，避免冷启动影响
- **多次迭代** - 统计学稳定的结果
- **多Batch Size** - 测试扩展性
- **详细报告** - 延迟、吞吐量、加速比

查看 [BENCHMARK_GUIDE.md](BENCHMARK_GUIDE.md) 获取完整文档。

## 📝 文档

| 文档 | 说明 |
|------|------|
| [QUICK_START.md](QUICK_START.md) | 快速开始指南 |
| [BENCHMARK_GUIDE.md](BENCHMARK_GUIDE.md) | 性能测试详细文档 |

## 🔧 实现细节

### 关键修改的文件

#### 1. fused_mirage_ops.py
Mirage融合算子的核心实现：
- `FusedRMSNormQKV` - 参数引用实现
- `set_use_mirage()` - 全局开关
- `is_mirage_enabled()` - 状态查询

#### 2. qwen3.py
Qwen3模型的Mirage集成：
- `Qwen3Attention` - 始终创建qkv_proj以支持权重加载
- `Qwen3DecoderLayer` - 条件创建fused_norm_qkv
- 正确处理residual连接避免双重归一化

### 关键设计决策

1. **参数引用而非参数创建** - 避免权重加载冲突
2. **始终实例化原始层** - 保证AutoWeightsLoader正常工作
3. **PyTorch fallback实现** - 为未来Mirage kernel提供接口
4. **torch.compile兼容** - 避免logger调用和shape unpacking

## 📈 预期性能

基于参数引用实现：
- **小batch (1-2)**: 性能相当或略慢（函数调用开销）
- **中batch (4-8)**: 开始显示优势
- **大batch (16+)**: 优势更明显

**注意：** 当前为PyTorch fallback实现。实际Mirage kernel实现后，性能将显著提升。

## 🎯 当前状态

- ✅ 成功集成到vLLM
- ✅ 权重加载正常
- ✅ 推理功能完整
- ✅ 测试框架完善
- ⏳ Mirage kernel实现（当前为PyTorch fallback）

## 🔍 技术要求

### 环境依赖
- CUDA 12.x
- PyTorch 2.8.0+
- Triton 3.1.0
- vLLM (latest)
- Mirage

### GPU要求
- 建议: 16GB+ VRAM
- 当前配置针对16GB GPU优化：
  - `max_model_len=16`
  - `enforce_eager=True` (禁用CUDA graphs)
  - `gpu_memory_utilization=0.85`

## 🐛 故障排除

### OOM错误
- 减小 `--batch-sizes`
- 降低 `max_model_len` (在test_mirage_qwen3.py中)
- 增加warmup迭代数

### 性能波动
- 增加 `--num-warmup`（例如10或20）
- 增加 `--num-iterations`（例如10或20）
- 关闭其他GPU进程

详见 [BENCHMARK_GUIDE.md](BENCHMARK_GUIDE.md) 的故障排除部分。

## 🤝 贡献

欢迎提交Issues和Pull Requests！

## 📄 许可

Apache License 2.0

## 🙏 致谢

- [vLLM](https://github.com/vllm-project/vllm) - 高性能LLM推理引擎
- [Mirage](https://github.com/mirage-project/mirage) - 超优化的深度学习编译器

---

**关键创新：** 参数引用模式 (Parameter Reference Pattern) 解决了融合算子与vLLM权重加载系统的兼容性问题。
