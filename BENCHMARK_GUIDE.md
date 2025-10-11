# Mirage Performance Benchmark Guide

## 文件说明

1. **test_prompts.txt** - 包含120+个测试prompts的文件
   - 短prompts (1-10 tokens)
   - 中等prompts (10-30 tokens)
   - 长prompts (30-60 tokens)
   - 代码生成prompts
   - 问答prompts
   - 创意写作prompts
   - 技术解释prompts
   - 推理逻辑prompts
   - 多语言prompts
   - 边界情况测试

2. **benchmark_mirage.py** - 全面的性能测试脚本
   - 自动warmup（预热）
   - 多次迭代求平均
   - 多种batch size测试
   - 详细统计分析
   - 自动生成对比报告

## 快速开始

### 基础用法（测试所有batch sizes）

```bash
# 设置环境变量
export PATH=/usr/local/cuda-12.4/bin:$PATH
export PYTHONPATH=/root/clean-mirage/vllm:$PYTHONPATH

# 运行完整benchmark（baseline + mirage）
python benchmark_mirage.py \
    --prompts test_prompts.txt \
    --max-tokens 50 \
    --num-warmup 5 \
    --num-iterations 5 \
    --batch-sizes 1 2 4 8
```

### 只测试Mirage（更快）

```bash
python benchmark_mirage.py --skip-baseline
```

### 只测试Baseline

```bash
python benchmark_mirage.py --skip-mirage
```

### 自定义配置

```bash
python benchmark_mirage.py \
    --model Qwen/Qwen3-0.6B \
    --prompts test_prompts.txt \
    --max-tokens 100 \
    --temperature 0.7 \
    --top-p 0.9 \
    --num-warmup 10 \
    --num-iterations 10 \
    --batch-sizes 1 4 8 16 \
    --output-report my_report.txt \
    --output-json my_results.json \
    --verbose
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | Qwen/Qwen3-0.6B | 模型名称或路径 |
| `--prompts` | test_prompts.txt | Prompts文件路径 |
| `--max-tokens` | 50 | 每个prompt生成的最大token数 |
| `--temperature` | 0.8 | 采样温度 |
| `--top-p` | 0.95 | Top-p采样参数 |
| `--num-warmup` | 5 | Warmup迭代次数（预热，不计入统计） |
| `--num-iterations` | 5 | 每个batch size的benchmark迭代次数 |
| `--batch-sizes` | 1 2 4 8 | 要测试的batch sizes列表 |
| `--skip-baseline` | False | 跳过baseline测试 |
| `--skip-mirage` | False | 跳过Mirage测试 |
| `--output-report` | benchmark_report.txt | 输出报告文件 |
| `--output-json` | benchmark_results.json | 输出JSON结果文件 |
| `--verbose` | False | 显示详细日志 |

## Warmup的重要性

脚本自动处理warmup，原因：

1. **GPU预热** - 首次运行时GPU需要初始化
2. **缓存预热** - vLLM的KV cache需要预热
3. **编译优化** - PyTorch的JIT编译需要预热
4. **稳定性能** - 前几次运行通常不稳定

默认5次warmup迭代，不计入性能统计。

## 输出文件

### 1. benchmark_report.txt（可读报告）

```
====================================================================================================
MIRAGE PERFORMANCE BENCHMARK REPORT
====================================================================================================

Generated: 2025-10-11 04:30:00

CONFIGURATION
----------------------------------------------------------------------------------------------------
Model: Qwen/Qwen3-0.6B
Max tokens: 50
Temperature: 0.8
Top-p: 0.95
Warmup iterations: 5
Benchmark iterations: 5

DETAILED RESULTS BY BATCH SIZE
----------------------------------------------------------------------------------------------------

Batch Size: 1
  ----------------------------------------------------------------------------------------------------
  LATENCY (ms):
    Baseline:   150.23 ±  5.12  (min:   143.45, max:   158.32, median:   149.87)
    Mirage:     125.67 ±  4.23  (min:   120.12, max:   131.45, median:   124.98)
    Speedup:    1.196x  (-16.4% latency reduction)
  THROUGHPUT (tokens/s):
    Baseline:    33.25 ±  1.13
    Mirage:      39.78 ±  1.34
    Speedup:     1.196x

...

====================================================================================================
SUMMARY
====================================================================================================

Average Latency Speedup:    1.215x
Average Throughput Speedup: 1.215x

✅ Mirage is 1.22x faster than baseline!
```

### 2. benchmark_results.json（详细数据）

包含所有原始数据，可用于进一步分析：
- 每次迭代的延迟
- 每次迭代的吞吐量
- 所有batch size的完整统计
- 配置信息

## 性能指标说明

1. **Latency (延迟)** - 完成推理所需时间（毫秒）
   - 越低越好
   - 提供均值、标准差、最小值、最大值、中位数

2. **Throughput (吞吐量)** - 每秒生成的token数
   - 越高越好
   - 提供均值和标准差

3. **Speedup (加速比)** - Mirage相对于baseline的性能提升
   - > 1.0 表示Mirage更快
   - < 1.0 表示baseline更快
   - ≈ 1.0 表示性能相当

## 最佳实践

1. **充足的warmup** - 至少5次，复杂模型建议10次
2. **多次迭代** - 至少5次，获得稳定的统计结果
3. **多个batch size** - 测试1, 2, 4, 8等，观察扩展性
4. **固定随机种子** - 保证结果可复现
5. **关闭其他GPU进程** - 避免干扰

## 示例：完整性能测试

```bash
# 1. 设置环境
export PATH=/usr/local/cuda-12.4/bin:$PATH
export PYTHONPATH=/root/clean-mirage/vllm:$PYTHONPATH

# 2. 运行完整benchmark
python benchmark_mirage.py \
    --prompts test_prompts.txt \
    --max-tokens 50 \
    --num-warmup 10 \
    --num-iterations 10 \
    --batch-sizes 1 2 4 8 16 \
    --output-report detailed_report.txt \
    --output-json detailed_results.json \
    --verbose

# 3. 查看结果
cat detailed_report.txt
```

## 故障排除

### OOM错误
- 减小 `--batch-sizes`（例如只测试 1 2 4）
- 减小 `--max-tokens`
- 在test script中降低max_model_len

### 性能波动大
- 增加 `--num-warmup`（例如10或20）
- 增加 `--num-iterations`（例如10或20）
- 关闭其他GPU进程

### 测试太慢
- 使用 `--skip-baseline` 只测Mirage
- 减少batch sizes数量
- 减少迭代次数

## 预期结果

基于parameter-reference实现，预期：
- **小batch (1-2)**: Mirage可能略慢或相当（由于额外的函数调用开销）
- **中batch (4-8)**: Mirage应该开始显示优势
- **大batch (16+)**: Mirage优势更明显（实际Mirage kernel实现后）

注意：当前实现是PyTorch fallback，真正的Mirage kernel实现后性能会显著提升。
