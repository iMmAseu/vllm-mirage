# Quick Start: Mirage + vLLM for Qwen3

## 环境设置

每次运行前需要设置以下环境变量：

```bash
export PATH=/usr/local/cuda-12.4/bin:$PATH
export PYTHONPATH=/root/clean-mirage/vllm:$PYTHONPATH
export VLLM_USE_MIRAGE=1  # 启用Mirage融合算子
```

## 快速测试

### 基础功能测试

```bash
# 只测试Mirage（跳过baseline，更快）
python test_mirage_qwen3.py --skip-baseline --verbose
```

### 性能对比测试

```bash
# 完整性能benchmark（baseline vs Mirage）
python benchmark_mirage.py \
    --prompts test_prompts.txt \
    --max-tokens 50 \
    --num-warmup 5 \
    --num-iterations 5 \
    --batch-sizes 1 2 4
```

## 主要文件说明

| 文件 | 用途 |
|------|------|
| `test_mirage_qwen3.py` | 基础功能测试 |
| `benchmark_mirage.py` | 完整性能测试 |
| `test_prompts.txt` | 测试prompts数据集（120+ prompts） |
| `BENCHMARK_GUIDE.md` | 详细的benchmark使用文档 |

## 核心实现

### Mirage融合算子架构

使用**参数引用模式**避免权重加载问题：

```python
# FusedRMSNormQKV 不创建新参数，而是引用现有层
class FusedRMSNormQKV(nn.Module):
    def __init__(self, norm_layer, qkv_proj, ...):
        self.norm_layer = norm_layer  # 引用现有RMSNorm
        self.qkv_proj = qkv_proj      # 引用现有QKV projection

    def forward(self, hidden_states):
        normed = self.norm_layer(hidden_states)
        qkv, _ = self.qkv_proj(normed)
        return qkv
```

### 关键修改的文件

1. **vllm/vllm/model_executor/layers/fused_mirage_ops.py**
   - FusedRMSNormQKV实现（参数引用模式）

2. **vllm/vllm/model_executor/models/qwen3.py**
   - Qwen3模型集成Mirage算子
   - 始终创建qkv_proj以支持权重加载

## 预期性能

- ✅ 权重加载成功（无KeyError）
- ✅ 模型初始化成功
- ✅ 推理功能正常
- ⏱️ 性能提升取决于batch size（当前为PyTorch fallback，实际Mirage kernel实现后会更快）

## 故障排除

### OOM错误
在 `test_mirage_qwen3.py` 中调整：
- `max_model_len=16`（当前设置）
- `enforce_eager=True`（禁用CUDA graphs）
- `gpu_memory_utilization=0.85`

### 详细文档
查看 `BENCHMARK_GUIDE.md` 获取完整的性能测试指南。
