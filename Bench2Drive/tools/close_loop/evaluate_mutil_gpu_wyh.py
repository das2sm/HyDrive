import os
import subprocess
import time
import signal
import sys
import json
sys.path.append('/home/wuyanhao/WorkSpace/SparseDrive')
from typing import Dict, List, Optional
import re
from leaderboard.scripts.process_manager import *
from carla_tools.split_xml import genereate_xml_from_json

from time import sleep

route_info_path = '/home/wuyanhao/WorkSpace/SparseDrive/leaderboard/data'
full_route_path = '/home/wuyanhao/WorkSpace/SparseDrive/leaderboard/data/bench2drive220.xml'
base_command = "bash leaderboard/scripts/run_evaluation.sh"
num_gpu = 8
used_gpu = [0, 1, 2, 3, 4, 5, 6, 7]
port_list_0 = [30000, 30150, 30300, 30450, 30600, 30750, 30900, 31050, 31200, 31350]
port_list_1 = [50000, 50150, 50300, 50450, 50600, 50750, 50900, 51050, 51200, 51350]
# traj_path = "/home/wuyanhao/WorkSpace/SparseDrive/leaderboard/data/bench2drive220_0_unified_framework_0417_traj.xml"
base_route_path = "leaderboard/data"
# route_version = "Pedestrain_Collision"
route_version = "bench2drive220_"
config_path = "projects_sp/configs/e2e_b2d.py"
weights_path = "/home/wuyanhao/data/weights/e2e_b2d_20250622-175205.078979"
control_version = '5m2hz' # or '2m5hz
base_save_path = "output/"
base_save_path = base_save_path + '_' + control_version
base_result_path = os.path.join(base_save_path, "result")
base_log_path = os.path.join(base_save_path, "log")
visulization_path = os.path.join(base_save_path, "visulization")

os.makedirs(base_save_path, exist_ok=True)
os.makedirs(base_result_path, exist_ok=True)
os.makedirs(base_log_path, exist_ok=True)
os.makedirs(visulization_path, exist_ok=True)
nick_name_pretext = 'eval_'
if control_version == '5m2hz':
    controler_paranter = '/home/wuyanhao/WorkSpace/SparseDrive/leaderboard/pid/best_5m_2hz.json'
elif control_version == '2m5hz':
    controler_paranter = '/home/wuyanhao/WorkSpace/SparseDrive/leaderboard/pid/best_2m_5hz.json'

auto_check=True

def build_command():
    commands = []
    for i in range(num_gpu):
        gpu = str(used_gpu[i])
        command_i = base_command + " " + str(port_list_0[i]) + " " + str(port_list_1[i]) + " True" # 构建基础命令
        command_i = command_i + " " + os.path.join(base_route_path, route_version + str(i) + ".xml") # 添加traj路径
        command_i = command_i + " " + "team_code/sparsedrive_b2d_agent.py" # 添加agent路径
        command_i = command_i + " " + config_path + "+" + weights_path # 添加config路径和weights路径
        command_i = command_i + " " + os.path.join(base_result_path, str(i) + ".json") # 添加save_path路径
        command_i = command_i + " " + visulization_path # 可视化路径
        command_i = command_i + " traj " + gpu +" "+ controler_paranter
        command_i = command_i + " 2>&1 | tee "  # 重定向以及
        command_i = command_i + os.path.join(base_log_path, str(i)+'.log')
        commands.append(command_i)
    command_dict = {}
    for i in range(num_gpu):
        command_dict[nick_name_pretext+str(i)] = commands[i]
    return command_dict
command_dict = build_command()

manager = ScreenManager(base_save_path)

# 启动所有仿真模型
for name, cmd in command_dict.items():
    manager.start_command(name, cmd)

# 保存会话信息到文件（供 killer.py 读取）
with open(f"{base_save_path}/sessions.json", "w") as f:
    json.dump(list(command_dict.keys()), f)

print("仿真模型已启动！")


# 提取命令信息（包含日志路径和screen名称）
commands_info = []
for i, cmd in enumerate(command_dict):
    cmd = command_dict[nick_name_pretext+str(i)]
    screen_name = f"eval_{i}"
    log_path = os.path.join(base_log_path, str(i)+'.log')
    commands_info.append({
        "raw_command": cmd,
        "screen_name": screen_name,
        "log_path": log_path
    })

print(commands_info)

# 错误检测函数
def contains_errors(log_path):
    error_keywords = [
        "Watchdog exception",
        "AttributeError",
        "Timeout",
        "RuntimeError",
        "61.0",
        # "failed",
        "False",
        "fault",
        # "cost time",
        "time-out",
        "60000ms",
        "UnboundLocalError",
    ]
  
    try:
        with open(log_path, "r") as f:
            # 高效读取最后10行
            lines = []
            for line in reversed(f.readlines()):
                lines.append(line)
                if len(lines) >= 20:
                    break
            content = "".join(reversed(lines))
          
            return any(kw in content for kw in error_keywords)
    except Exception as e:
        print(f"⚠️ 检查日志错误 [{log_path}]: {str(e)}")
        return False

if auto_check:
    print("开始自动监控...")
    sleep(20)
    # 守护进程循环
    while True:
        print("\n" + "="*50)
        print(f"开始新一轮检查 ({time.ctime()})")
    
        for info in commands_info:
            if not info["log_path"]:
                continue
            
            print(f"检查 {info['screen_name']}...", end=" ")
        
            # if contains_errors(info["log_path"]):
            if contains_errors(info["log_path"]):
                print("检测到错误！尝试重启...")
            
                # 终止现有screen会话
                kill_args = ["sh", "/home/gaoliang/workspace/GL/E2E_Model/carla_tools/clean_carla_for_run_scene.sh", info["screen_name"]]
                subprocess.run(kill_args,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
            
                # 重启命令
                try:
                    subprocess.run(
                        ["screen", "-dmS", info["screen_name"], "bash", "-c", info["raw_command"]],
                        check=True
                    )
                    print(f"🔄 重启成功 {info['screen_name']}")
                    # 清空旧日志
                    open(info["log_path"], "w").close()
                except subprocess.CalledProcessError as e:
                    print(f"🔥 重启失败 {info['screen_name']}: {e}")
            else:
                print("运行正常")
    
        print("="*50 + "\n")
        time.sleep(60)  # 每次检查间隔60秒
