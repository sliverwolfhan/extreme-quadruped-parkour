#!/usr/bin/env python3
import subprocess
import time
import re
import sys
import datetime

def run_cmd(cmd):
    try:
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
        return res.stdout
    except Exception as e:
        return ""

def main():
    print("[Watchdog] Starting Real-time ATDog2 Training Monitor...", flush=True)
    consecutive_errors = 0
    
    while True:
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        
        # 1. Check if training process is alive in container
        ps_out = run_cmd("docker exec robot-lab ps aux | grep -E 'train.py' | grep -v grep")
        if not ps_out.strip():
            print(f"[{timestamp}] [WARNING] train.py process not detected in docker!", flush=True)
            consecutive_errors += 1
            if consecutive_errors > 3:
                print(f"[{timestamp}] [FATAL] Process appears dead. Exiting watchdog.", flush=True)
                break
            time.sleep(15)
            continue
        
        consecutive_errors = 0
        
        # 2. Grab recent log output
        log_tail = run_cmd("docker exec robot-lab tail -n 50 /workspace/logs/rsl_rl/train_atdog2_meihua.log")
        
        iter_match = re.search(r"Learning iteration (\d+)/(\d+)", log_tail)
        reward_match = re.search(r"Mean reward:\s+([-\d.]+)", log_tail)
        val_loss_match = re.search(r"Mean value_function loss:\s+([-\d.]+)", log_tail)
        eplen_match = re.search(r"Mean episode length:\s+([-\d.]+)", log_tail)
        terrain_match = re.search(r"Curriculum/mean_terrain_level:\s+([-\d.]+)", log_tail)
        speed_match = re.search(r"Computation:\s+(\d+)\s+steps/s", log_tail)
        eta_match = re.search(r"ETA:\s+([:\d]+)", log_tail)
        
        cur_iter = iter_match.group(1) if iter_match else "?"
        max_iter = iter_match.group(2) if iter_match else "3000"
        reward = float(reward_match.group(1)) if reward_match else None
        val_loss = float(val_loss_match.group(1)) if val_loss_match else None
        eplen = float(eplen_match.group(1)) if eplen_match else None
        terrain = float(terrain_match.group(1)) if terrain_match else None
        speed = speed_match.group(1) if speed_match else "?"
        eta = eta_match.group(1) if eta_match else "?"
        
        # 3. Check GPU status
        gpu_out = run_cmd("nvidia-smi --query-gpu=temperature.gpu,memory.used,memory.total,power.draw --format=csv,noheader,nounits")
        gpu_info = gpu_out.strip().split(",") if gpu_out.strip() else ["?", "?", "?", "?"]
        gpu_temp = gpu_info[0].strip() if len(gpu_info) > 0 else "?"
        gpu_mem = gpu_info[1].strip() if len(gpu_info) > 1 else "?"
        gpu_pwr = gpu_info[3].strip() if len(gpu_info) > 3 else "?"
        
        # 4. Anomaly detection
        status = "HEALTHY"
        if val_loss is not None and (val_loss > 100.0 or val_loss < -100.0):
            status = f"ANOMALY (Value Loss={val_loss:.1f})"
        elif reward is not None and reward < -1000.0:
            status = f"ANOMALY (Reward={reward:.1f})"
        
        # Print structured status
        print(
            f"[{timestamp}] Iter: {cur_iter}/{max_iter} | "
            f"Reward: {f'{reward:+.2f}' if reward is not None else '?':>7} | "
            f"ValLoss: {f'{val_loss:.4f}' if val_loss is not None else '?':>7} | "
            f"EpLen: {f'{eplen:.1f}' if eplen is not None else '?':>5} | "
            f"Terrain: {f'{terrain:.2f}' if terrain is not None else '?'}/1.0 | "
            f"Speed: {speed} st/s | "
            f"GPU: {gpu_temp}°C {gpu_mem}MB {gpu_pwr}W | "
            f"ETA: {eta} | [{status}]",
            flush=True
        )
        
        if "ANOMALY" in status:
            print(f"[{timestamp}] [ALERT] Anomaly detected! Taking intervention...", flush=True)
            # Kill runaway training if necessary
            run_cmd("docker exec robot-lab ps aux | grep -E 'train.py' | grep -v grep | awk '{print $2}' | xargs -r docker exec robot-lab kill -9")
            print(f"[{timestamp}] Runaway training terminated by watchdog. Exiting.", flush=True)
            break
            
        if iter_match and int(cur_iter) >= int(max_iter):
            print(f"[{timestamp}] [DONE] Reached target {max_iter} iterations successfully!", flush=True)
            break
            
        time.sleep(15)

if __name__ == "__main__":
    main()
