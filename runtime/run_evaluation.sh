#!/bin/bash
set -euo pipefail
mkdir -p close_loop_log/log
mkdir -p close_loop_log/routes
mkdir -p close_loop_log/result

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export F2D_DIR="${F2D_DIR:-$ROOT/Fail2Drive}"
export CARLA_ROOT="${CARLA_ROOT:-$ROOT/f2d_carla}"
export LEADERBOARD_ROOT="${LEADERBOARD_ROOT:-$F2D_DIR/fail2drive_leaderboard}"
export SCENARIO_RUNNER_ROOT="${SCENARIO_RUNNER_ROOT:-$F2D_DIR/fail2drive_scenario_runner}"

export PYTHONPATH="$LEADERBOARD_ROOT:$SCENARIO_RUNNER_ROOT:$ROOT/leaderboard:$ROOT:$CARLA_ROOT/PythonAPI/carla:${PYTHONPATH:-}"

export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh

export CHALLENGE_TRACK_CODENAME=SENSORS
export DEBUG_CHALLENGE=0
export REPETITIONS=1 # multiple evaluation runs
export RESUME="${RESUME:-False}"
export TRAFFIC_MANAGER_SEED="${TRAFFIC_MANAGER_SEED:-0}"

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

RESUME_ARG="--no-resume"
case "${RESUME,,}" in
true|1|yes) RESUME_ARG="--resume" ;;
false|0|no) ;;
*)
    echo "ERROR: RESUME must be one of True/False/1/0/yes/no, got: $RESUME" >&2
    exit 2
    ;;
esac

CAMPAIGN_PYTHON="${HYDRIVE_PYTHON_EXECUTABLE:-python}"

CUDA_VISIBLE_DEVICES="${GPU_RANK}" "${CAMPAIGN_PYTHON}" "${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py" \
--routes="${ROUTES}" \
--repetitions="${REPETITIONS}" \
--track="${CHALLENGE_TRACK_CODENAME}" \
--checkpoint="${CHECKPOINT_ENDPOINT}" \
--agent="${TEAM_AGENT}" \
--agent-config="${TEAM_CONFIG}" \
--debug="${DEBUG_CHALLENGE}" \
--record="${RECORD_PATH:-}" \
"${RESUME_ARG}" \
--traffic-manager-seed="${TRAFFIC_MANAGER_SEED}" \
--port="${PORT}" \
--traffic-manager-port="${TM_PORT}"
