#!/bin/bash
# Disaggregated Prefill+Decode 批量测试运行脚本

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 获取脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 默认参数
PROXY_URL="http://localhost:8000"
PREFILL_URL="http://localhost:8100"
DECODE_URL="http://localhost:8200"
OUTPUT_DIR="disagg_batch_results"

# 显示使用帮助
show_help() {
    echo "Disaggregated Prefill+Decode 批量测试脚本"
    echo ""
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  --batch-size N        批次大小（必需）"
    echo "  --requests N          总请求数（默认: 8）"
    echo "  --max-tokens N        最大生成token数（默认: 128）"
    echo "  --test-length L       测试序列长度: 1024, 2048, 4096, 8192（默认: 2048）"
    echo "  --preset P            预设配置: short, medium, long"
    echo "  --proxy-url URL       代理服务器URL（默认: http://localhost:8000）"
    echo "  --output-dir DIR      输出目录（默认: disagg_batch_results）"
    echo "  -h, --help            显示此帮助信息"
    echo ""
    echo "预设配置:"
    echo "  short   - 1024 tokens输入, 64 tokens输出, 8个请求"
    echo "  medium  - 2048 tokens输入, 128 tokens输出, 8个请求"
    echo "  long    - 4096 tokens输入, 256 tokens输出, 4个请求"
    echo ""
    echo "示例:"
    echo "  # 使用medium预设配置，批次大小为4"
    echo "  $0 --preset medium --batch-size 4"
    echo ""
    echo "  # 自定义参数"
    echo "  $0 --batch-size 2 --requests 4 --max-tokens 64 --test-length 2048"
    echo ""
}

# 检查服务是否运行
check_services() {
    echo -e "${BLUE}检查服务状态...${NC}"
    
    # 检查代理服务器
    if curl -s "$PROXY_URL/health" > /dev/null 2>&1; then
        echo -e "  ${GREEN}✓${NC} 代理服务器运行正常"
    else
        echo -e "  ${RED}✗${NC} 代理服务器未运行"
        echo -e "${YELLOW}请先启动服务: ./test_single_machine.sh all${NC}"
        exit 1
    fi
    
    # 检查Prefill服务器
    if curl -s "$PREFILL_URL/health" > /dev/null 2>&1; then
        echo -e "  ${GREEN}✓${NC} Prefill服务器运行正常"
    else
        echo -e "  ${YELLOW}⚠${NC} Prefill服务器状态未知"
    fi
    
    # 检查Decode服务器
    if curl -s "$DECODE_URL/health" > /dev/null 2>&1; then
        echo -e "  ${GREEN}✓${NC} Decode服务器运行正常"
    else
        echo -e "  ${YELLOW}⚠${NC} Decode服务器状态未知"
    fi
    
    echo ""
}

# 解析命令行参数
BATCH_SIZE=""
REQUESTS=""
MAX_TOKENS=""
TEST_LENGTH=""
PRESET=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --requests)
            REQUESTS="$2"
            shift 2
            ;;
        --max-tokens)
            MAX_TOKENS="$2"
            shift 2
            ;;
        --test-length)
            TEST_LENGTH="$2"
            shift 2
            ;;
        --preset)
            PRESET="$2"
            shift 2
            ;;
        --proxy-url)
            PROXY_URL="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo -e "${RED}错误: 未知选项 $1${NC}"
            show_help
            exit 1
            ;;
    esac
done

# 检查必需参数
if [ -z "$BATCH_SIZE" ] && [ -z "$PRESET" ]; then
    echo -e "${RED}错误: 必须指定 --batch-size 或 --preset${NC}"
    echo ""
    show_help
    exit 1
fi

# 检查服务
check_services

# 构建命令
CMD="python3 $SCRIPT_DIR/simple_batch_test.py"

if [ -n "$PRESET" ]; then
    CMD="$CMD --preset $PRESET"
fi

if [ -n "$BATCH_SIZE" ]; then
    CMD="$CMD --batch-size $BATCH_SIZE"
fi

if [ -n "$REQUESTS" ]; then
    CMD="$CMD --requests $REQUESTS"
fi

if [ -n "$MAX_TOKENS" ]; then
    CMD="$CMD --max-tokens $MAX_TOKENS"
fi

if [ -n "$TEST_LENGTH" ]; then
    CMD="$CMD --test-length $TEST_LENGTH"
fi

CMD="$CMD --proxy-url $PROXY_URL"
CMD="$CMD --prefill-url $PREFILL_URL"
CMD="$CMD --decode-url $DECODE_URL"
CMD="$CMD --output-dir $OUTPUT_DIR"

# 显示配置信息
echo -e "${BLUE}开始批量测试...${NC}"
echo -e "架构: ${GREEN}Disaggregated Prefill+Decode${NC}"
if [ -n "$PRESET" ]; then
    echo -e "预设: ${GREEN}$PRESET${NC}"
fi
if [ -n "$BATCH_SIZE" ]; then
    echo -e "批次大小: ${GREEN}$BATCH_SIZE${NC}"
fi
echo ""

# 执行测试
echo -e "${BLUE}执行命令:${NC}"
echo "$CMD"
echo ""

eval $CMD

# 显示结果位置
echo ""
echo -e "${GREEN}测试完成！${NC}"
echo -e "结果保存在: ${YELLOW}$OUTPUT_DIR/${NC}"

