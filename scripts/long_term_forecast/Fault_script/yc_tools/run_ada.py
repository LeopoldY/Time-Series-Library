import os
import glob
import subprocess
import time
import re
import csv
import sys

# ================= 配置 =================
ADA_RUN_DIR = "/mnt/sdc1/skx/Time-Series-Library/scripts/long_term_forecast/Fault_script/ada_run/"
RESULT_CSV = "final_experiment_results.csv"
# =======================================

def main():
    if not os.path.exists(ADA_RUN_DIR):
        print(f"Error: Directory {ADA_RUN_DIR} not found.")
        return

    scripts = sorted(glob.glob(os.path.join(ADA_RUN_DIR, "*.sh")))
    print(f"Found {len(scripts)} scripts. Results will be saved to '{RESULT_CSV}'.")

    # 初始化 CSV 文件
    if not os.path.exists(RESULT_CSV):
        with open(RESULT_CSV, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Device', 'Model', 'Seq_Len', 'MSE', 'MAE'])

    # 正则表达式
    # 匹配 utils.py 生成的标记: >>> EXP_INFO: Device:xx Model:xx Seq:xx
    info_pattern = re.compile(r">>> EXP_INFO: Device:(.*?) Model:(.*?) Seq:(\d+)")
    # 匹配 run.py 的输出: mse:0.123, mae:0.456
    metrics_pattern = re.compile(r"mse:([\d\.]+),\s*mae:([\d\.]+)")

    total_start = time.time()

    for i, script in enumerate(scripts):
        print(f"\n[{i+1}/{len(scripts)}] Executing: {os.path.basename(script)}")
        
        # 当前实验上下文
        current_context = {} 
        
        try:
            # 启动子进程，实时捕获输出
            process = subprocess.Popen(
                ["bash", script],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1 # 行缓冲
            )

            # 逐行读取输出
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                
                if line:
                    line = line.strip()
                    print(f"    | {line}") # 打印到屏幕以便监控

                    # 1. 捕获实验配置信息
                    info_match = info_pattern.search(line)
                    if info_match:
                        current_context = {
                            'Device': info_match.group(1).strip(),
                            'Model': info_match.group(2).strip(),
                            'Seq_Len': info_match.group(3).strip()
                        }
                    
                    # 2. 捕获实验结果
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
                                mae
                            ])
                        print(f"    [SAVE] >> Recorded: {current_context['Device']} (Len={current_context['Seq_Len']}) MSE={mse}")

            if process.returncode != 0:
                print(f"    [WARN] Script finished with error code {process.returncode}")

        except Exception as e:
            print(f"    [ERROR] Execution failed: {e}")

    print("\n" + "="*50)
    print(f"Batch Execution Completed. Total time: {(time.time()-total_start)/60:.2f} min")
    print(f"All data saved to: {os.path.abspath(RESULT_CSV)}")
    print("="*50)

if __name__ == "__main__":
    main()