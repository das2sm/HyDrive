import os
import time
import glob
import json
import xml.etree.ElementTree as ET
from argparse import Namespace

import socket
import subprocess
import psutil
import atexit
import signal
import numpy as np

import warnings
warnings.filterwarnings("ignore")

from tools.close_loop.merge_route_json import merge_route_json
from tools.close_loop.custom_metric import get_metric
from tools.close_loop.ability_benchmark import main as get_ability
from tools.close_loop.efficiency_smoothness_benchmark import read_from_json, seg_compute_comfort_metric
from tools.close_loop.collect_video_mp_with_metric import _process_one_folder, images_to_video_parallel

CARLA_ROOT = os.environ["CARLA_ROOT"]
BASE_PORT = 30000
BASE_TM_PORT = 50000
IS_BENCH2DRIVE = True
TEAM_AGENT = "team_code/sparsedrive_b2d_agent.py"
MODEL_LIST = [
    "projects/configs/sparsedrive_stage2.py+ckpt/sparsedrive_small_b2d_stage2.pth",
]

## b2d220
GPU_RANK_LIST = [0, 1, 2, 3, 4, 5, 6, 7]
CARLA_GPU_RANK_LIST = [0, 1, 2, 3, 4, 5, 6, 7]

SUBSET_LEN = 220
BASE_ROUTES = "leaderboard/data/bench2drive220"

CHALLENGE_TRACK_CODENAME = "SENSORS"
TASK_NUM = len(GPU_RANK_LIST)


def get_scene_name(case):
    return case.findall("scenarios")[0].findall("scenario")[0].attrib["name"]

def sort_root(cases):
    cases.sort(key=get_scene_name)
    scene_names = [get_scene_name(x) for x in cases]
    unique_scene_names = set(scene_names)
    repetition = len(scene_names) // len(unique_scene_names)
    new_cases = []
    for i in range(repetition):
        repetition_case = cases[i::repetition]
        new_cases.extend(repetition_case)
    return new_cases

def split_list_into_n_parts(lst, n):
    k, m = divmod(len(lst), n)
    return (lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

def split_xml(base_route, task_num, subset_len):
    tree = ET.parse(f'{base_route}.xml')
    root = tree.getroot()
    case = root.findall('route')
    case = sort_root(case)
    case = case[:subset_len]
    new_root = ET.Element("routes")
    for x in case:
        new_root.append(x)
    new_tree = ET.ElementTree(new_root)
    new_tree.write(f'close_loop_log/routes/bench2drive{subset_len}.xml', encoding='utf-8', xml_declaration=True)

    results = split_list_into_n_parts(case, task_num)
    for index, re in enumerate(results):
        new_root = ET.Element("routes")
        for x in re:
            new_root.append(x)
        new_tree = ET.ElementTree(new_root)
        new_tree.write(f'close_loop_log/routes/bench2drive{subset_len}_{index}.xml', encoding='utf-8', xml_declaration=True)

def update_xml(base_route, task_num, subset_len):
    completed = []
    incompleted = []
    for index in range(task_num):
        tree = ET.parse(f'close_loop_log/routes/bench2drive{subset_len}_{index}.xml')
        root = tree.getroot()
        case = root.findall('route')
        endpoint = f"close_loop_log/result/bench2drive_{index}.json"
        with open(endpoint) as fd:
            data = json.load(fd)
        completed_route_ids = [x["route_id"].split("_")[1] for x in data["_checkpoint"]["records"]]
        completed_case = [x for x in case if x.attrib["id"] in completed_route_ids]
        incompleted_case = [x for x in case if x.attrib["id"] not in completed_route_ids]
        print("index:", index, "len:", len(case))
        print("completed:", [x.attrib["id"] for x in completed_case])
        print("incompleted:", [x.attrib["id"] for x in incompleted_case])
        completed.append(completed_case)
        incompleted.extend(incompleted_case)

    incompleted = split_list_into_n_parts(incompleted, task_num)
    for index, incompleted_case in enumerate(incompleted):
        completed_case = completed[index]
        new_root = ET.Element("routes")
        for x in completed_case:
            new_root.append(x)
        for x in incompleted_case:
            new_root.append(x)
        new_tree = ET.ElementTree(new_root)
        print("index:", index, "len:", len(new_root))
        new_tree.write(f'close_loop_log/routes/bench2drive{subset_len}_{index}.xml', encoding='utf-8', xml_declaration=True)

def check_xml_exist(base_route, task_num, subset_len):
    for index in range(task_num):
        exist = os.path.exists(f'close_loop_log/routes/bench2drive{subset_len}_{index}.xml')
        if not exist:
            return False
    
    return True

def find_free_port(starting_port):
    port = starting_port
    while True:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("localhost", port))
                return port
        except OSError:
            port += 1

def start_process(command):
    process = subprocess.Popen(command, shell=True, preexec_fn=os.setsid)
    # atexit.register(os.killpg, process.pid, signal.SIGKILL)
    return process

def find_pid_by_command(command):
    try:
        result = subprocess.check_output(['pgrep', '-f', command])
        pids = [int(pid) for pid in result.decode('utf-8').split('\n') if pid.isdigit()]
        return pids
    except subprocess.CalledProcessError:
        return []

def is_process_running(pids):
    running = False
    for pid in pids:
        try:
            process = psutil.Process(pid)
            process.wait(timeout=0)
            running = False
        except psutil.NoSuchProcess:
            running = False
        except psutil.AccessDenied:
            running = False
        else:
            running = True
    return running

def kill_processes(pids):
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
            print(f"Terminated process with PID: {pid}")
        except ProcessLookupError:
            print(f"Process with PID {pid} does not exist.")
        except PermissionError:
            print(f"Permission denied to terminate process with PID {pid}.")
        except Exception as e:
            print(f"Failed to terminate process with PID {pid}: {e}")

def check_finished(i, task_num, subset_len):
    endpoint = f"close_loop_log/result/bench2drive_{i}.json"
    if not os.path.exists(endpoint):
        return False
    with open(endpoint) as fd:
        try:
            data = json.load(fd)
            progress = data["_checkpoint"]["progress"]
            if progress[0] == progress[1]:
                return True
            else:
                return False
        except:
            return False

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
            lines = []
            for line in reversed(f.readlines()):
                lines.append(line)
                if len(lines) >= 10:
                    break
            content = "".join(reversed(lines))
          
            return any(kw in content for kw in error_keywords)
    except Exception as e:
        print(f"⚠️ 检查日志错误 [{log_path}]: {str(e)}")
        return False

def append_file(src_path: str, dst_path: str) -> None:
    """
    将 src_path 文件的全部内容追加到 dst_path 文件末尾。
    若 dst_path 不存在会自动创建。
    """
    with open(src_path, 'rb') as fin, open(dst_path, 'ab') as fout:
        fout.write(fin.read())

def test_one_model(MODEL_NAME):
    os.makedirs("close_loop_log/log", exist_ok=True)
    os.makedirs("close_loop_log/routes", exist_ok=True)
    os.makedirs("close_loop_log/result", exist_ok=True)
    split_xml(BASE_ROUTES, TASK_NUM, SUBSET_LEN)
    
    TEAM_CONFIG = MODEL_NAME

    def get_command(i):
        port = BASE_PORT + i * 150
        port = find_free_port(port)
        tm_port = BASE_TM_PORT + i * 150
        route = f"close_loop_log/routes/bench2drive{SUBSET_LEN}_{i}.xml"
        checkpoint_endpoint = f"close_loop_log/result/bench2drive_{i}.json"
        gpu_rank = GPU_RANK_LIST[i]
        carla_gpu_rank = CARLA_GPU_RANK_LIST[i]

        python_command = f'''
            export CARLA_ROOT={CARLA_ROOT}
            export CARLA_SERVER=$CARLA_ROOT/CarlaUE4.sh
            export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI
            export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla
            export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg
            export PYTHONPATH=$PYTHONPATH:leaderboard
            export PYTHONPATH=$PYTHONPATH:leaderboard/team_code
            export PYTHONPATH=$PYTHONPATH:scenario_runner
            export SCENARIO_RUNNER_ROOT=scenario_runner
            export IS_BENCH2DRIVE=True
            python3 leaderboard/leaderboard/leaderboard_evaluator.py \\
                --routes={route} \\
                --repetitions=1 \\
                --track=SENSORS \\
                --checkpoint={checkpoint_endpoint} \\
                --agent={TEAM_AGENT} \\
                --agent-config={TEAM_CONFIG} \\
                --debug=0 \\
                --record= \\
                --resume=True \\
                --port={port} \\
                --traffic-manager-port={tm_port} \\
                --gpu-rank={gpu_rank} >> close_loop_log/log/b2d_python_{str(i).zfill(2)}.log 2>&1 
        '''

        carla_command = f"{CARLA_ROOT}/CarlaUE4.sh -RenderOffScreen -nosound -carla-rpc-port={port} -graphicsadapter={carla_gpu_rank} >> close_loop_log/log/b2d_carla_{str(i).zfill(2)}.log"# 2>&1"
        
        return python_command, carla_command, port

    ports = dict()
    finished = np.zeros((TASK_NUM), dtype=np.bool)

    for i in range(TASK_NUM):
        finished[i] = check_finished(i, TASK_NUM, SUBSET_LEN)
        if finished[i]:
            continue
        python_command, carla_command, port = get_command(i)
        ports[i] = port
        print(f"Staring process {i}")
        start_process(python_command)
        start_process(carla_command)
        time.sleep(5)
    
    time.sleep(10)

    while not finished.all():
        for i in range(TASK_NUM):
            finished[i] = check_finished(i, TASK_NUM, SUBSET_LEN)
            if finished[i]:
                print(f"Porcess {i} finished.")
                python_pids = find_pid_by_command(f"b2d_python_{str(i).zfill(2)}")
                carla_pids = find_pid_by_command(f"carla-rpc-port={ports[i]}")
                kill_processes(python_pids)
                kill_processes(carla_pids)
                continue

            python_pids = find_pid_by_command(f"b2d_python_{str(i).zfill(2)}")
            python_running = len(python_pids) == 1
            carla_pids = find_pid_by_command(f"carla-rpc-port={ports[i]}")
            carla_running = len(carla_pids) == 3

            print("Process: ", i)
            print("Python pids: ", python_pids)
            print("Python running: ", python_running)
            print("Carla pids: ", carla_pids)
            print("Carla running: ", carla_running)
            print("Port: ", ports[i])

            if not (python_running and carla_running):
                print(f"Killing process {i}")
                kill_processes(python_pids)
                kill_processes(carla_pids)
                time.sleep(60)
                print(f"Restarting process {i}")
                python_command, carla_command, port = get_command(i)
                ports[i] = port
                start_process(python_command)
                start_process(carla_command)
                time.sleep(20)
            
            time.sleep(5)

    time.sleep(10)
    ## metrics ##
    # driving score
    res = merge_route_json("close_loop_log/result")
    with open("close_loop_log/results.json", 'w') as file:
        json.dump(res, file, indent=4)

    res = get_metric("close_loop_log/result/merged.json")
    with open("close_loop_log/metric.json", 'w') as file:
        json.dump(res, file, indent=4)

    images_to_video_parallel(result_dir="close_loop_log", max_workers=TASK_NUM)

    os.system("ps aux |grep CarlaUE4 |grep -v grep | awk '{print $2}' | xargs -r kill -9")
    os.system("ps aux |grep leaderboard_evaluator.py |grep -v grep | awk '{print $2}' | xargs -r kill -9")

if __name__ == '__main__':
    for model_name in MODEL_LIST:
        os.system("ps aux |grep CarlaUE4 |grep -v grep | awk '{print $2}' | xargs -r kill -9")
        os.system("ps aux |grep leaderboard_evaluator.py |grep -v grep | awk '{print $2}' | xargs -r kill -9")

        test_one_model(model_name)
        model_tag = model_name.split("/")[1]
        os.system(f"mv close_loop_log/videos close_loop_log/videos_{model_tag}_route_{SUBSET_LEN}")
        os.system(f"mv close_loop_log/ close_loop_logs_save_new/close_loop_log_{model_tag}_route_{SUBSET_LEN}")
        print(f"End of evaluation of {model_tag}")