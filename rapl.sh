# 第一次读取
val1=$(sudo cat /sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj)

# 等待一段时间（比如1秒）
sleep 1

# 第二次读取
val2=$(sudo cat /sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj)

# 计算功率（瓦特）
echo "功率: $(echo "scale=2; ($val2 - $val1) / 1000000" | bc) W"