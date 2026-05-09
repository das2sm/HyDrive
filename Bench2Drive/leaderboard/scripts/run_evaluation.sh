#!/bin/bash
mkdir -p close_loop_log/log
mkdir -p close_loop_log/routes
mkdir -p close_loop_log/result

# Fail2Drive environment variables
export F2D_DIR=/home/ace428/Soham/HyDrive/Bench2Drive/Fail2Drive
export CARLA_ROOT=/home/ace428/Soham/HyDrive/Bench2Drive/f2d_carla
export LEADERBOARD_ROOT=$F2D_DIR/fail2drive_leaderboard
export SCENARIO_RUNNER_ROOT=$F2D_DIR/fail2drive_scenario_runner

export PYTHONPATH=/home/ace428/Soham/HyDrive/Bench2Drive/leaderboard:/home/ace428/Soham/HyDrive/Bench2Drive:$CARLA_ROOT/PythonAPI/carla:$LEADERBOARD_ROOT:$SCENARIO_RUNNER_ROOT:$PYTHONPATH

export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh

export CHALLENGE_TRACK_CODENAME=SENSORS
export DEBUG_CHALLENGE=0
export REPETITIONS=1 # multiple evaluation runs
export RESUME=True

export PORT=$1
export TM_PORT=$2
export IS_BENCH2DRIVE=$3
export ROUTES=$4
export TEAM_AGENT=$5
export TEAM_CONFIG=$6
export CHECKPOINT_ENDPOINT=$7
export SAVE_PATH=$8
export PLANNER_TYPE=$9
export GPU_RANK=${10}

CUDA_VISIBLE_DEVICES=${GPU_RANK} python ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py \
--routes=${ROUTES} \
--repetitions=${REPETITIONS} \
--track=${CHALLENGE_TRACK_CODENAME} \
--checkpoint=${CHECKPOINT_ENDPOINT} \
--agent=${TEAM_AGENT} \
--agent-config=${TEAM_CONFIG} \
--debug=${DEBUG_CHALLENGE} \
--record=${RECORD_PATH} \
--resume=${RESUME} \
--port=${PORT} \
--traffic-manager-port=${TM_PORT} \
