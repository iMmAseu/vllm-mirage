#!/bin/bash

echo "======================================================================"
echo "           Mirage + vLLM 安装验证脚本"
echo "======================================================================"
echo ""

# Test 1: Mirage
echo "[1/4] 检查 Mirage 库..."
python -c "import mirage; print('  ✅ Mirage 已安装')" 2>/dev/null || {
    echo "  ❌ Mirage 未安装"
    echo "     解决方法: cd /root/clean-mirage/mirage && pip install -e ."
    exit 1
}

# Test 2: vLLM
echo "[2/4] 检查 vLLM 库..."
python -c "import vllm; print('  ✅ vLLM 已安装')" 2>/dev/null || {
    echo "  ❌ vLLM 未安装"
    echo "     解决方法: cd /root/clean-mirage/vllm && pip install -e ."
    exit 1
}

# Test 3: 融合算子
echo "[3/4] 检查 Mirage 融合算子..."
python -c "from vllm.model_executor.layers.fused_mirage_ops import FusedRMSNormQKV; print('  ✅ 融合算子可导入')" 2>/dev/null || {
    echo "  ❌ 融合算子不可用"
    echo "     可能原因: vLLM 未正确安装或代码未同步"
    exit 1
}

# Test 4: Qwen3 修改
echo "[4/4] 检查 Qwen3 模型修改..."
python -c "from vllm.model_executor.models.qwen3 import _MIRAGE_FUSED_OPS_AVAILABLE; print(f'  ✅ Qwen3 已集成 Mirage (算子可用: {_MIRAGE_FUSED_OPS_AVAILABLE})')" 2>/dev/null || {
    echo "  ❌ Qwen3 修改不可用"
    echo "     可能原因: vLLM 未正确安装或代码未同步"
    exit 1
}

echo ""
echo "======================================================================"
echo "✅ 所有检查通过！Mirage + vLLM 已正确安装"
echo "======================================================================"
echo ""
echo "下一步："
echo "  1. 启用 Mirage: export VLLM_USE_MIRAGE=1"
echo "  2. 运行测试: python test_mirage_qwen3.py --verbose"
echo ""
