#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

# NOTE: Replace with your paths
export F2D_DIR=/home/ace428/Soham/HyDrive/Bench2Drive/Fail2Drive
export CARLA_ROOT=/home/ace428/Soham/HyDrive/Bench2Drive/f2d_carla

export LEADERBOARD_ROOT=$F2D_DIR/fail2drive_leaderboard
export SCENARIO_RUNNER_ROOT=$F2D_DIR/fail2drive_scenario_runner

# NOTE: The PYTHONPATH used by your project may already include leaderboard and scenario_runner paths. It's best if these are removed.
export PYTHONPATH=$CARLA_ROOT/PythonAPI/carla:$F2D_DIR/fail2drive_leaderboard:$F2D_DIR/fail2drive_scenario_runner:$PYTHONPATH
export PYTHONPATH=/home/ace428/Soham/HyDrive/Bench2Drive/leaderboard:/home/ace428/Soham/HyDrive/Bench2Drive:$CARLA_ROOT/PythonAPI/carla:$LEADERBOARD_ROOT:$SCENARIO_RUNNER_ROOT:$PYTHONPATH

BASE_PORT=3001
BASE_TM_PORT=5001
IS_BENCH2DRIVE=True
BASE_ROUTES=$F2D_DIR/fail2drive_split/Generalization_Animals_1082
# BASE_ROUTES=/home/ace428/Soham/HyDrive/Bench2Drive/leaderboard/data/bench2drive220
TEAM_AGENT=leaderboard/team_code/sparsedrive_b2d_agent_occ.py
TEAM_CONFIG=projects/configs/sparsedrive_stage2.py+ckpt/sparsedrive_small_b2d_stage2.pth+f2d_eval
BASE_CHECKPOINT_ENDPOINT=close_loop_log/result/bench2drive
SAVE_PATH=close_loop_log/result/save
PLANNER_TYPE=only_traj
GPU_RANK=0

PORT=$BASE_PORT
TM_PORT=$BASE_TM_PORT
ROUTES="${BASE_ROUTES}.xml"
CHECKPOINT_ENDPOINT="${BASE_CHECKPOINT_ENDPOINT}.json"

bash leaderboard/scripts/run_evaluation.sh $PORT $TM_PORT $IS_BENCH2DRIVE $ROUTES $TEAM_AGENT $TEAM_CONFIG $CHECKPOINT_ENDPOINT $SAVE_PATH $PLANNER_TYPE $GPU_RANK

