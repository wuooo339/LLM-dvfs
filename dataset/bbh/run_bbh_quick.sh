#!/bin/bash

# BBH数据集快速获取工具运行脚本

echo "🧠 BBH数据集快速获取工具"
echo "========================"
echo ""

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 未安装"
    exit 1
fi

# 检查datasets库
echo "🔍 检查依赖..."
if ! python3 -c "import datasets" 2>/dev/null; then
    echo "⚠️  datasets库未安装"
    echo "📦 正在安装datasets库..."
    pip install datasets
    if [ $? -ne 0 ]; then
        echo "❌ datasets库安装失败"
        echo "请手动安装: pip install datasets"
        exit 1
    fi
    echo "✅ datasets库安装成功"
else
    echo "✅ datasets库已安装"
fi

echo ""
echo "📖 使用说明:"
echo "  ./run_bbh_quick.sh --list                                    # 列出所有任务"
echo "  ./run_bbh_quick.sh --task date_understanding --count 3       # 获取3条日期理解任务"
echo "  ./run_bbh_quick.sh --task logical_deduction_three_objects --count 1 --format prompt-only  # 只输出prompt"
echo "  ./run_bbh_quick.sh --task causal_judgment --count 5 --output results.json --format json   # 保存为JSON"
echo ""

# 运行Python脚本，传递所有参数
python3 get_bbh_quick.py "$@"
