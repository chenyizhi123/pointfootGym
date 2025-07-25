#!/bin/bash
# PointFoot 机器人评估示例脚本

echo "🚀 PointFoot 机器人性能评估 - 示例运行"
echo "=========================================="

# 设置机器人类型
export ROBOT_TYPE=PF_TRON1A
echo "✅ 设置机器人类型: $ROBOT_TYPE"

# 检查是否有训练好的模型
LOGS_DIR="logs"
if [ ! -d "$LOGS_DIR" ]; then
    echo "❌ 错误: 未找到logs目录，请先训练模型"
    echo "运行训练命令: python legged_gym/scripts/train.py --task=pointfoot_rough --headless"
    exit 1
fi

echo "📁 查找已训练的模型..."
LATEST_MODEL=$(ls -t logs/ | head -n 1)
if [ -z "$LATEST_MODEL" ]; then
    echo "❌ 错误: 未找到训练好的模型"
    exit 1
fi

echo "🤖 找到最新模型: $LATEST_MODEL"

echo ""
echo "🎯 开始性能评估..."
echo "评估内容:"
echo "  1. 追踪指令精度 (30%权重)"
echo "  2. 地形跨越性能 (25%权重)"  
echo "  3. 行走稳定性 (25%权重)"
echo "  4. 抗外力干扰能力 (20%权重)"
echo ""

# 运行评估
python legged_gym/scripts/play_evaluate.py --task=pointfoot_rough

echo ""
echo "✅ 评估完成！"
echo "📊 查看详细报告: logs/evaluation_report.json" 