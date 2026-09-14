import os
import glob
import subprocess
import time
import re
import csv
import sys

# ================= 配置区域 =================
# 1. 脚本所在目录 (由 AFD.py 生成的 .sh 文件存放处)
ADA_RUN_DIR = "/mnt/sdc1/skx/Time-Series-Library/scripts/long_term_forecast/Fault_script/ada_run/"

# 2. 结果保存路径 (您可以修改为您想要的文件名)
RESULT_CSV = "specific_devices_results_raw_data.csv"

# 3. 指定要训练的设备列表 (白名单)
# 逻辑：只要脚本文件名中包含以下任意字符串，就会被执行。
# 示例: ["NE40E_设备83", "S9300"]
# 如果留空 []，脚本将不会运行任何东西 (为了安全，防止误跑所有)
TARGET_DEVICES = [
    "设备83",
    "设备69",
    "设备58",
    "设备27"
]
# ===========================================

def main():
    print("="*50)
    print("Running SAFE Experiment for Specific Devices")
    print("="*50)

    if not os.path.exists(ADA_RUN_DIR):
        print(f"[Error] Directory {ADA_RUN_DIR} not found.")
        return

    # 1. 获取所有脚本
    all_scripts = sorted(glob.glob(os.path.join(ADA_RUN_DIR, "*.sh")))
    
    # 2. 筛选目标脚本
    target_scripts = []
    if not TARGET_DEVICES:
        print("[Warning] TARGET_DEVICES list is empty. No scripts will be run.")
        print("Please edit the script and add device names to TARGET_DEVICES.")
        return

    print(f"Filtering scripts for targets: {TARGET_DEVICES} ...")
    
    for script in all_scripts:
        script_name = os.path.basename(script)
        # 检查文件名是否包含任一目标设备名
        for target in TARGET_DEVICES:
            target_script_name = target + "_For_Machine_Learning.sh"
            if target_script_name in script_name:
                target_scripts.append(script)
                break
    
    if not target_scripts:
        print("[Error] No matching scripts found for the specified devices.")
        return

    print(f"Found {len(target_scripts)} matching scripts out of {len(all_scripts)} total.")
    print(f"Results will be saved to: {RESULT_CSV}\n")

    # 3. 初始化 CSV 文件
    # 如果文件不存在，写入表头；如果存在，追加模式
    file_exists = os.path.exists(RESULT_CSV)
    with open(RESULT_CSV, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Device', 'Model', 'Seq_Len', 'MSE', 'MAE', 'Timestamp'])

    # 正则表达式 (匹配 utils.py 的埋点)
    info_pattern = re.compile(r">>> EXP_INFO: Device:(.*?) Model:(.*?) Seq:(\d+)")
    metrics_pattern = re.compile(r"mse:([\d\.]+),\s*mae:([\d\.]+)")

    total_start = time.time()

    # 4. 循环执行
    for i, script in enumerate(target_scripts):
        script_name = os.path.basename(script)
        print(f"[{i+1}/{len(target_scripts)}] Executing: {script_name}")
        
        current_context = {} 
        
        try:
            # 启动子进程
            process = subprocess.Popen(
                ["bash", script],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )

            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                
                if line:
                    line = line.strip()
                    # 仅打印关键信息以保持控制台整洁，或者打印所有 line 用于调试
                    # print(f"    | {line}") 

                    # A. 捕获配置信息
                    info_match = info_pattern.search(line)
                    if info_match:
                        current_context = {
                            'Device': info_match.group(1).strip(),
                            'Model': info_match.group(2).strip(),
                            'Seq_Len': info_match.group(3).strip()
                        }
                        print(f"    -> Configuration: {current_context}")
                    
                    # B. 捕获结果指标
                    metrics_match = metrics_pattern.search(line)
                    if metrics_match and current_context:
                        mse = metrics_match.group(1)
                        mae = metrics_match.group(2)
                        
                        # 写入 CSV
                        with open(RESULT_CSV, 'a', newline='', encoding='utf-8') as f:
                            writer = csv.writer(f)
                            writer.writerow([
                                current_context['Device'],
                                current_context['Model'],
                                current_context['Seq_Len'],
                                mse,
                                mae,
                                time.strftime("%Y-%m-%d %H:%M:%S")
                            ])
                        print(f"    [SAVE] >> Recorded: Len={current_context['Seq_Len']}, MSE={mse}")

            if process.returncode != 0:
                print(f"    [WARN] Script finished with error code {process.returncode}")

        except Exception as e:
            print(f"    [ERROR] Execution failed: {e}")
        
        print("-" * 50)

    print("\n" + "="*50)
    print(f"Specific Execution Completed. Total time: {(time.time()-total_start)/60:.2f} min")
    print(f"Check results in: {os.path.abspath(RESULT_CSV)}")
    print("="*50)

if __name__ == "__main__":
    main()