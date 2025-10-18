#!/usr/bin/env python3
"""
CPU监控模块
基于perf工具实现CPU功耗监控
使用sudo perf stat -e power/energy-pkg/ -a sleep 0.1获取功耗数据
"""

import subprocess
import time
import threading
import json
import getpass
from typing import Dict, Any, List
from datetime import datetime

class CPUMonitor:
    """CPU监控器 - 仅使用perf stat监控CPU功耗"""
    
    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self.monitoring = False
        self.data = []
        self.monitor_thread = None
        self.running = True
        
        # 存储上一次的能量值和时间，用于计算功耗
        self._last_energy_j = None
        self._last_time = None
        
        # 传感器与PkgWatt采样缓存，避免高频外部命令导致卡顿
        self._last_temp_result = None
        self._last_temp_time = 0.0
        # 将间隔设为0表示每次采样都重新获取，保证CSV按采样频率刷新
        self._temp_update_interval = 0.0  # 秒
        
        self._last_pkgwatt_result = None
        self._last_pkgwatt_time = 0.0
        self._pkgwatt_update_interval = 0.1  # 秒

        # 异步更新线程（避免阻塞采样循环）
        self._temp_thread = threading.Thread(target=self._temp_updater_loop, daemon=True)
        self._pkg_thread = threading.Thread(target=self._pkgwatt_updater_loop, daemon=True)
        self._temp_thread.start()
        self._pkg_thread.start()
        
        print("CPU监控器初始化完成，使用perf stat监控CPU功耗")
    
    def _run_sudo_perf(self, cmd: List[str]) -> subprocess.CompletedProcess:
        """使用sudo运行perf命令"""
        try:
            # 获取sudo密码
            if not hasattr(self, '_sudo_password'):
                print("需要sudo权限来运行perf命令以获取CPU功耗数据")
                self._sudo_password = getpass.getpass("请输入sudo密码: ")
            
            # 构建正确的sudo命令
            if cmd[0] == 'sudo':
                # 如果命令已经包含sudo，直接使用
                sudo_cmd = cmd
            else:
                # 如果命令不包含sudo，添加sudo
                sudo_cmd = ['sudo', '-S'] + cmd
            
            process = subprocess.Popen(
                sudo_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            stdout, stderr = process.communicate(input=self._sudo_password + '\n', timeout=15)
            
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=process.returncode,
                stdout=stdout,
                stderr=stderr
            )
            
        except Exception as e:
            print(f"sudo perf执行失败: {e}")
            return subprocess.CompletedProcess(args=cmd, returncode=1, stdout="", stderr=str(e))
    
    def get_cpu_power(self) -> Dict[str, Any]:
        """使用sudo perf stat -e power/energy-pkg/ -a sleep 0.1获取CPU功耗"""
        cpu_info = {
            'timestamp': time.time(),
            'datetime': datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            'method': 'perf_stat'
        }
        
        try:
            # 使用sudo perf stat命令获取CPU功耗数据
            cmd = ['sudo', '-S', 'perf', 'stat', '-e', 'power/energy-pkg/', '-a', 'sleep', '0.1']
            
            # 如果还没有sudo密码，使用预设密码
            if not hasattr(self, '_sudo_password'):
                self._sudo_password = "hpc"
            
            # 使用Popen来提供密码输入
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # 发送密码并等待命令完成
            stdout, stderr = process.communicate(input=self._sudo_password + '\n', timeout=15)
            
            result = subprocess.CompletedProcess(
                args=cmd,
                returncode=process.returncode,
                stdout=stdout,
                stderr=stderr
            )
            
            if result.returncode == 0:
                # 解析perf输出，查找Joules数据和时间
                lines = result.stderr.strip().split('\n')
                energy_joules = None
                elapsed_time = None
                
                for line in lines:
                    line = line.strip()
                    
                    # 查找包含Joules和power/energy-pkg/的行
                    # 格式示例: "26.66 Joules power/energy-pkg/"
                    if 'Joules' in line and 'power/energy-pkg/' in line:
                        try:
                            # 提取Joules数值
                            parts = line.split()
                            if len(parts) >= 3:
                                joules_str = parts[0].replace(',', '')
                                energy_joules = float(joules_str)
                        except Exception as e:
                            print(f"解析Joules数据失败: {e}")
                    
                    # 查找时间信息
                    # 格式示例: "0.103197762 seconds time elapsed"
                    elif 'seconds time elapsed' in line:
                        try:
                            parts = line.split()
                            if len(parts) >= 3:
                                time_str = parts[0]
                                elapsed_time = float(time_str)
                        except Exception as e:
                            print(f"解析时间数据失败: {e}")
                
                # 如果找到了能量和时间数据，计算功率
                if energy_joules is not None and elapsed_time is not None:
                    cpu_info['cpu_energy_j'] = energy_joules
                    # 功率 = 能量增量 / 实际测量时间
                    power_watts = energy_joules / elapsed_time
                    cpu_info['cpu_power_w'] = max(0, power_watts)
            else:
                print(f"perf stat命令失败: {result.stderr}")
                
        except Exception as e:
            print(f"获取CPU功耗数据失败: {e}")
        
        return cpu_info

    def _parse_sensors_output(self, output: str) -> Dict[str, Any]:
        """从`sensors`命令输出解析CPU温度。

        适配常见AMD平台k10temp输出，如:
        k10temp-pci-00c3
        Adapter: PCI adapter
        Tctl:         +38.0°C
        Tccd1:        +37.8°C
        ...
        解析策略:
        - 优先取首个k10temp分节中的Tctl作为cpu_temperature_c
        - 若无Tctl，取同节所有Tccd*的平均值
        - 回退: 在全局范围内查找包含 "temp1" 的行数字
        """
        cpu_temp_c = None
        per_socket_temps = []  # 顺序即插槽编号
        lines = output.splitlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith('k10temp'):
                j = i + 1
                tctl = None
                tccd_vals = []
                while j < len(lines) and lines[j].strip() != '':
                    s = lines[j].strip()
                    if s.startswith('Tctl:'):
                        try:
                            val = s.split(':', 1)[1]
                            val = val.replace('°C', '').replace('+', '').strip()
                            tctl = float(val)
                        except Exception:
                            pass
                    elif s.startswith('Tccd') and ':' in s:
                        try:
                            val = s.split(':', 1)[1]
                            val = val.replace('°C', '').replace('+', '').strip()
                            tccd_vals.append(float(val))
                        except Exception:
                            pass
                    j += 1
                if tctl is not None:
                    per_socket_temps.append(tctl)
                elif tccd_vals:
                    per_socket_temps.append(sum(tccd_vals) / len(tccd_vals))
                i = j
            else:
                i += 1

        # 若未解析到k10temp，退回全局temp1
        if not per_socket_temps:
            for s in lines:
                st = s.strip()
                if st.startswith('temp1:') and '°C' in st:
                    try:
                        val = st.split(':', 1)[1]
                        val = val.replace('°C', '').replace('+', '').strip()
                        per_socket_temps.append(float(val))
                        break
                    except Exception:
                        continue

        if per_socket_temps:
            cpu_temp_c = sum(per_socket_temps) / len(per_socket_temps)

        result = { 'cpu_temperature_c': cpu_temp_c if cpu_temp_c is not None else 0.0 }
        # 只展开前两个插槽温度（Socket 0和1）
        for idx, t in enumerate(per_socket_temps[:2]):
            result[f'cpu_socket{idx}_temperature_c'] = t
        result['cpu_socket_count'] = min(len(per_socket_temps), 2)
        return result

    def get_cpu_temperature(self) -> Dict[str, Any]:
        """返回最近一次温度缓存（由后台线程更新）。"""
        return self._last_temp_result or { 'cpu_temperature_c': 0.0 }

    def _temp_updater_loop(self):
        """后台线程：周期性刷新温度缓存。"""
        while True:
            now = time.time()
            try:
                result = subprocess.run(['sensors'], capture_output=True, text=True, timeout=2)
                if result.returncode == 0:
                    self._last_temp_result = self._parse_sensors_output(result.stdout)
                else:
                    self._last_temp_result = { 'cpu_temperature_c': 0.0 }
            except Exception:
                self._last_temp_result = { 'cpu_temperature_c': 0.0 }
            self._last_temp_time = now
            # 保持与采样间隔一致（或使用配置的更新间隔）
            time.sleep(max(0.05, self._temp_update_interval or 0.1))

    def get_cpu_pkgwatt_per_socket(self) -> Dict[str, Any]:
        """返回最近一次PkgWatt缓存（由后台线程更新）。

        实现策略: 运行 `sudo turbostat --interval 0.1 --quiet --show PkgWatt` 短暂采样，
        在超短时间内读取stdout中的浮点行，将其依次映射到socket0, socket1,...
        如失败则返回0。
        """
        return self._last_pkgwatt_result or {
            'cpu_socket0_pkg_watt': 0.0,
            'cpu_socket1_pkg_watt': 0.0,
            'cpu_pkgwatt_sum': 0.0
        }

    def _pkgwatt_updater_loop(self):
        """后台线程：周期性刷新PkgWatt缓存。"""
        while True:
            now = time.time()
            fields: Dict[str, Any] = {}
            try:
                if not hasattr(self, '_sudo_password'):
                    self._sudo_password = "hpc"

                cmd = ['sudo', '-S', 'turbostat', '--interval', '0.1', '--num_iterations', '1', '--quiet', '--show', 'PkgWatt']
                result = subprocess.run(
                    cmd,
                    input=self._sudo_password + '\n',
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                combined = (result.stdout or '') + "\n" + (result.stderr or '')

                watts = []
                for line in combined.splitlines():
                    s = line.strip()
                    if not s:
                        continue
                    try:
                        val = float(s)
                        watts.append(val)
                    except Exception:
                        continue

                total_val = None
                socket_vals = []
                if len(watts) == 3:
                    total_val = watts[0]
                    socket_vals = watts[1:3]
                elif len(watts) >= 2:
                    socket_vals = watts[-2:]
                    total_val = sum(socket_vals)
                elif len(watts) == 1:
                    total_val = watts[0]

                # 只记录前两个Socket的PkgWatt数据
                for idx, w in enumerate(socket_vals[:2]):
                    fields[f'cpu_socket{idx}_pkg_watt'] = max(0.0, w)
                if total_val is not None:
                    fields['cpu_pkgwatt_total'] = max(0.0, total_val)
                if socket_vals:
                    fields['cpu_pkgwatt_sum'] = sum(socket_vals[:2])
            except Exception:
                pass

            if not fields:
                fields = {
                    'cpu_socket0_pkg_watt': 0.0,
                    'cpu_socket1_pkg_watt': 0.0,
                    'cpu_pkgwatt_sum': 0.0
                }
            self._last_pkgwatt_result = fields
            self._last_pkgwatt_time = now
            time.sleep(max(0.05, self._pkgwatt_update_interval or 0.1))
    
    def get_cpu_info(self) -> Dict[str, Any]:
        """获取CPU信息: 功耗 + 温度 + 每插槽PkgWatt"""
        info = self.get_cpu_power()
        temp = self.get_cpu_temperature()
        pkg = self.get_cpu_pkgwatt_per_socket()
        info.update(temp)
        info.update(pkg)
        return info
    
    def _monitor_loop(self):
        """监控循环"""
        while self.monitoring and self.running:
            start_time = time.time()
            
            cpu_info = self.get_cpu_info()
            self.data.append(cpu_info)
            
            # 计算剩余时间，确保精确的采样间隔
            elapsed = time.time() - start_time
            sleep_time = max(0, self.interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def start_monitoring(self):
        """开始监控"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.data = []
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        print(f"开始CPU监控，采样间隔: {self.interval}s")
    
    def stop_monitoring(self):
        """停止监控"""
        if not self.monitoring:
            return
        
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
        print("停止CPU监控")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取监控统计信息"""
        if not self.data:
            return {}
        
        # 提取功耗相关指标
        cpu_powers = [d.get('cpu_power_w', 0) for d in self.data if d.get('cpu_power_w', 0) > 0]
        cpu_energies = [d.get('cpu_energy_j', 0) for d in self.data if d.get('cpu_energy_j', 0) > 0]
        
        stats = {
            "monitoring_duration": self.data[-1]['timestamp'] - self.data[0]['timestamp'] if len(self.data) > 1 else 0,
            "sample_count": len(self.data),
            "method": "perf_stat_power_energy_pkg"
        }
        
        if cpu_powers:
            stats["cpu_power"] = {
                "avg": sum(cpu_powers) / len(cpu_powers),
                "max": max(cpu_powers),
                "min": min(cpu_powers),
                "unit": "Watts"
            }
        
        if cpu_energies:
            stats["cpu_energy"] = {
                "total": max(cpu_energies) - min(cpu_energies) if len(cpu_energies) > 1 else 0,
                "latest": cpu_energies[-1],
                "unit": "Joules"
            }
        
        return stats
    
    def save_data(self, filename: str):
        """保存监控数据"""
        data_to_save = {
            "interval": self.interval,
            "method": "perf_stat_power_energy_pkg",
            "statistics": self.get_statistics(),
            "raw_data": self.data
        }
        
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=2)
        
        print(f"CPU功耗监控数据已保存到: {filename}")

def main():
    """测试CPU功耗监控功能"""
    print("CPU功耗监控模块测试")
    print("=" * 50)
    
    monitor = CPUMonitor(interval=0.5)
    
    try:
        print("开始监控5秒...")
        monitor.start_monitoring()
        time.sleep(5)
        monitor.stop_monitoring()
        
        # 显示统计信息
        stats = monitor.get_statistics()
        print("\n监控统计:")
        print(f"监控时长: {stats.get('monitoring_duration', 0):.1f}秒")
        print(f"采样次数: {stats.get('sample_count', 0)}")
        print(f"监控方法: {stats.get('method', 'N/A')}")
        
        if 'cpu_power' in stats:
            power_stats = stats['cpu_power']
            print(f"CPU功耗 - 平均: {power_stats['avg']:.2f}W, 最大: {power_stats['max']:.2f}W, 最小: {power_stats['min']:.2f}W")
        
        if 'cpu_energy' in stats:
            energy_stats = stats['cpu_energy']
            print(f"CPU能量 - 总消耗: {energy_stats['total']:.2f}J, 当前值: {energy_stats['latest']:.2f}J")
        
        # 保存数据
        output_file = f"cpu_power_monitor_test_{int(time.time())}.json"
        monitor.save_data(output_file)
        
    except KeyboardInterrupt:
        print("\n用户中断测试")
        monitor.stop_monitoring()

if __name__ == "__main__":
    main()
