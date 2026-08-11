# Closed-loop runtime setup

The paper has two reproduction paths:

- The paper statistics can be reproduced from the released route-level data
  without CARLA; see the main README.
- Re-running the closed-loop study requires SparseDriveV2, the Fail2Drive
  plugin, Fail2Drive's CARLA distribution, the SparseDriveV2 checkpoint and
  anchor files, and the HyDrive overlay described here.

A new CARLA campaign is a replication with fresh simulator executions. It
should not be expected to reproduce every route outcome bit-for-bit.

## Runtime layout

Use SparseDriveV2's `bench2drive` branch as the runtime root. That branch
already contains the SparseDriveV2 model and the Bench2Drive leaderboard
layout. Fail2Drive's plugin branch is then nested inside that root,
as recommended by Fail2Drive:

```text
hydrive-runtime/
|-- Fail2Drive/
|   |-- fail2drive_leaderboard/
|   |-- fail2drive_scenario_runner/
|   `-- fail2drive_split/
|-- f2d_carla/
|-- ckpt/
|-- data/kmeans/
|-- leaderboard/
|-- projects/
|-- scripts/
`-- tests/manifests/
```

The campaign runner expects these names and relative locations.

## 1. Clone HyDrive and set absolute paths

Choose the parent directory on the first line. The defaults below keep the
release checkout, combined runtime, and campaign output beside one another
under your home directory:

```bash
export HYDRIVE_WORKSPACE="$HOME/hydrive-reproduction"
mkdir -p "$HYDRIVE_WORKSPACE"
cd "$HYDRIVE_WORKSPACE"

git clone https://github.com/das2sm/HyDrive.git HyDrive

export HYDRIVE="$HYDRIVE_WORKSPACE/HyDrive"
export RUNTIME="$HYDRIVE_WORKSPACE/hydrive-runtime"
export OUTPUT="$HYDRIVE_WORKSPACE/campaign-output"

printf 'HYDRIVE=%s\nRUNTIME=%s\nOUTPUT=%s\n' \
  "$HYDRIVE" "$RUNTIME" "$OUTPUT"
test -f "$HYDRIVE/patches/sparsedrive_planner_outputs.patch" &&
test -f "$HYDRIVE/patches/fail2drive_campaign_controls.patch" &&
echo "HyDrive release paths verified"
```

All three variables are absolute paths. The two `test` commands must succeed
before continuing. In a new shell, redefine the four exported variables above;
do not clone the repositories again.

## Tested platform

The reported campaign used Linux x86-64, Python 3.8.20, CUDA 11.8, cuDNN
8.7, PyTorch 2.0.1+cu118, NumPy 1.24.4, and CARLA 0.9.15. It requires an
NVIDIA GPU. Reserve at least 10 GiB for the campaign output directory. The
complete runtime installation requires substantially more space because the
Fail2Drive CARLA distribution and Python environment are stored separately.

## 2. Create the SparseDriveV2 runtime root

The paper used the May 8 SparseDriveV2 `bench2drive` snapshot. Commit
`3f0326d` matches the configuration and model code used by the campaign:

```bash
git clone --branch bench2drive --single-branch \
  https://github.com/swc-17/SparseDriveV2.git "$RUNTIME"
cd "$RUNTIME"
git checkout -b hydrive-reproduction \
  3f0326dcd59f13d9fcae27c23a8c70994da3131d

git apply --check "$HYDRIVE/patches/sparsedrive_planner_outputs.patch" && \
git apply "$HYDRIVE/patches/sparsedrive_planner_outputs.patch"

grep -q -- 'traj_selected_mode_index' \
  projects/mmdet3d_plugin/models/motion/decoder.py &&
echo "SparseDriveV2 planner-output patch verified"
```

The SparseDriveV2 patch exposes the complete proposal tensor, post-rescore
scores, suppression mask, and selected mode index at the decoder output. It
does not change proposal generation, learned scores, rescoring, or the selected
trajectory.

## 3. Create the Python environment

Follow SparseDriveV2's environment and CUDA-operator build procedure:

```bash
# Do not layer Conda on top of an active Python venv.
if [ -n "${VIRTUAL_ENV:-}" ]; then deactivate; fi

conda create -n sparsedrive python=3.8 -y
conda activate sparsedrive

python -m pip install --upgrade pip
python -m pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 \
  --extra-index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirement.txt
python -m pip install numpy==1.24.4 carla==0.9.15 einops==0.6.1

# PyTorch wheels include CUDA runtime libraries but not the nvcc compiler.
# Install the matching toolkit inside this Conda environment.
conda install -y cuda -c nvidia/label/cuda-11.8.0
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

"$CUDA_HOME/bin/nvcc" --version
"$CUDA_HOME/bin/nvcc" --version | grep -q 'release 11\.8' || {
  echo "CUDA 11.8 compiler required; found: $("$CUDA_HOME/bin/nvcc" --version | tail -n 1)" >&2
  exit 1
}

cd projects/mmdet3d_plugin/ops
python setup.py develop
cd ../../..
```

An existing system installation of the CUDA 11.8 toolkit can be used instead:
skip the `conda install` command and set `CUDA_HOME` to that installation's
root before updating `PATH` and `LD_LIBRARY_PATH`. No system CUDA path is
assumed. A newer compatible NVIDIA driver is acceptable; the compiler selected
by `CUDA_HOME` must still be CUDA 11.8 because the pinned PyTorch wheel was
built for CUDA 11.8.

HyDrive runs the SparseDriveV2 agent and campaign scripts in the single
`sparsedrive` environment. Deactivate any Python virtual environment before
activating this Conda environment; the shell prompt should not show both
`(.venv)` and `(sparsedrive)`.

Verify the core imports and recorded versions:

```bash
python - <<'PY'
import carla
import einops
import mmcv
import numpy
import torch
import deformable_aggregation_ext
import deformable_aggregation_with_depth_ext

print("CARLA", getattr(carla, "__version__", "unknown"))
print("einops", einops.__version__)
print("MMCV", mmcv.__version__)
print("NumPy", numpy.__version__)
print("PyTorch", torch.__version__, "CUDA", torch.version.cuda)
print("SparseDriveV2 CUDA extensions: OK")
assert numpy.__version__ == "1.24.4"
assert torch.__version__ == "2.0.1+cu118"
assert torch.cuda.is_available()
PY
```

## 4. Download the SparseDriveV2 runtime assets

Closed-loop evaluation does not require the multi-terabyte Bench2Drive
training dataset or its annotation PKLs. It does require the released stage-2
checkpoint and the six anchor files loaded by the stage-2 configuration:

```bash
mkdir -p ckpt data/kmeans

wget -c https://huggingface.co/wenchaosun/SparseDriveV2/resolve/main/sparsedrive_small_b2d_stage2.pth \
  -O ckpt/sparsedrive_small_b2d_stage2.pth

for file in \
  kmeans_det_900.npy \
  kmeans_map_100.npy \
  kmeans_motion_6.npy \
  path_1m_pts_15_1024_b2d_new_ego.npy \
  vel_seq_K256_t30.npy
do
  wget -c "https://huggingface.co/wenchaosun/SparseDriveV2/resolve/main/${file}" \
    -O "data/kmeans/${file}"
done

# The checkpoint uses the six-waypoint Bench2Drive trajectory archive, not
# the eight-waypoint archive with the similar unsuffixed name.
wget -c \
  https://huggingface.co/wenchaosun/SparseDriveV2/resolve/main/trajectory_1024_256_b2d.npz \
  -O data/kmeans/trajectory_1024_256.npz
```

Verify the campaign checkpoint and configuration identities before continuing:

```bash
sha256sum -c <<'EOF'
f15c889b759734894e6ac906403df55a9973d7e60e3101da3ac71e9091f7b130  ckpt/sparsedrive_small_b2d_stage2.pth
c22709e7c064b5215227dde7ca13dc8a4b28ef943a818e5313146a7410b777c5  projects/configs/sparsedrive_stage2.py
7fb58fcc09a8f18eb7bfe4a1db1d6625b7666e70af3ddc6f3d4a265890eea693  data/kmeans/kmeans_det_900.npy
b1f05c3db01c100a3ec103b3adf58e29143f1fbd66f728da9ab34c4b6c862335  data/kmeans/kmeans_map_100.npy
e23decadfd7584ac91a70a7c5e1222be8f066ecbbe40fbd0d54afc85eb76bef9  data/kmeans/kmeans_motion_6.npy
521f0b537a0eeed2a80d9e4f441cb6ef3c1c606c027b9f7b511b2584c0aa54aa  data/kmeans/path_1m_pts_15_1024_b2d_new_ego.npy
5bc170d6ec03627ae568161d4b071ecc1bac593cb2e25304060580dcfff7f321  data/kmeans/trajectory_1024_256.npz
d1b8b38ff382f24e72f8036dc74a4dbdaef5f73844d8a87cc7d24fe861a5cf84  data/kmeans/vel_seq_K256_t30.npy
EOF
```

Expected output:

```text
ckpt/sparsedrive_small_b2d_stage2.pth: OK
projects/configs/sparsedrive_stage2.py: OK
data/kmeans/kmeans_det_900.npy: OK
data/kmeans/kmeans_map_100.npy: OK
data/kmeans/kmeans_motion_6.npy: OK
data/kmeans/path_1m_pts_15_1024_b2d_new_ego.npy: OK
data/kmeans/trajectory_1024_256.npz: OK
data/kmeans/vel_seq_K256_t30.npy: OK
```

Do not continue if any file reports `FAILED`.

## 5. Install Fail2Drive

Fail2Drive has since merged the scenario-resolution fix that was carried in
the original campaign checkout. Pin the merged plugin revision below and apply
only the remaining campaign controls.

```bash
git clone --branch plugin --single-branch \
  https://github.com/autonomousvision/fail2drive.git "$RUNTIME/Fail2Drive"
git -C "$RUNTIME/Fail2Drive" checkout -b hydrive-strict-eval \
  b2025d5a8dabe3fcfe1f1f259f4010b0a3c61cf3
git -C "$RUNTIME/Fail2Drive" apply --check \
  "$HYDRIVE/patches/fail2drive_campaign_controls.patch" && \
git -C "$RUNTIME/Fail2Drive" apply \
  "$HYDRIVE/patches/fail2drive_campaign_controls.patch"

git -C "$RUNTIME/Fail2Drive" add -A
git -C "$RUNTIME/Fail2Drive" commit -m "Apply HyDrive strict evaluation patch"

grep -q -- "--no-resume" \
  Fail2Drive/fail2drive_leaderboard/leaderboard/leaderboard_evaluator.py &&
grep -q -- "HYDRIVE_SCENARIO_INIT_OK" \
  Fail2Drive/fail2drive_leaderboard/leaderboard/scenarios/scenario_manager.py &&
echo "Fail2Drive campaign controls verified"
```

Both `grep` commands must exit successfully. If either fails, do not prepare a
campaign: the nested Fail2Drive checkout is still missing required controls.

The remaining patch gives the evaluator unambiguous `--resume`/`--no-resume`
flags, makes scenario initialization depend on simulation time in strict mode,
and repairs mutation-during-iteration bookkeeping. The merged upstream commit
already supplies the corrected scenario classes and filenames, so those fixes
are not duplicated here.

Download Fail2Drive's customized CARLA distribution beside `Fail2Drive/`:

```bash
mkdir -p "$RUNTIME/f2d_carla"
curl -L \
  https://huggingface.co/datasets/SimonGer/fail2drive/resolve/main/fail2drive_simulator.tar.gz \
  | tar -xz -C "$RUNTIME/f2d_carla"
chmod +x "$RUNTIME/f2d_carla/CarlaUE4.sh"
```

Do not substitute the stock Bench2Drive CARLA package. The Fail2Drive routes
use assets supplied by the Fail2Drive simulator archive.

## 6. Install the HyDrive overlay

Run the following from the combined runtime:

```bash
cd "$RUNTIME"

cp "$HYDRIVE"/intervention/*.py leaderboard/team_code/

mkdir -p scripts/multiseed campaigns/multiseed_v1 tests/manifests analysis
cp "$HYDRIVE"/runtime/*.py scripts/multiseed/
cp "$HYDRIVE"/runtime/run_evaluation.sh leaderboard/scripts/run_evaluation.sh
cp "$HYDRIVE"/analysis/multiseed_campaign.py analysis/multiseed_campaign.py
cp "$HYDRIVE"/configs/fail2drive_test_routes.txt \
  tests/manifests/fail2drive_test.txt
cp "$HYDRIVE"/configs/fail2drive_dev_routes.txt \
  tests/manifests/fail2drive_dev.txt
cp "$HYDRIVE"/configs/protocol.example.json \
  campaigns/multiseed_v1/protocol.json
```

Write the absolute runtime, output, and Conda-Python paths into the local
protocol:

```bash
export PROTOCOL="$RUNTIME/campaigns/multiseed_v1/protocol.json"
export PYTHON="$(python -c 'import sys; print(sys.executable)')"

python - <<'PY'
import json
import os
from pathlib import Path

path = Path(os.environ["PROTOCOL"])
protocol = json.loads(path.read_text(encoding="utf-8"))
protocol["repo_root"] = os.environ["RUNTIME"]
protocol["artifact_root"] = os.environ["OUTPUT"]
protocol["python_executable"] = os.environ["PYTHON"]
path.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")

print("repo_root:", protocol["repo_root"])
print("artifact_root:", protocol["artifact_root"])
print("python_executable:", protocol["python_executable"])
PY
```

All three printed paths must begin with `/`. No manual JSON path replacement
is required.

Do not change the arms, route manifest, seeds, retry rules, or statistical
settings when reconstructing the reported protocol.

## 7. Record a clean runtime checkout

Campaign preparation intentionally refuses dirty source trees. Keep the large
simulator and nested Fail2Drive repository out of the parent repository's
status, then commit the overlay and local protocol:

```bash
printf '%s\n' 'Fail2Drive/' 'f2d_carla/' >> .git/info/exclude

git add leaderboard/team_code leaderboard/scripts/run_evaluation.sh \
  projects/mmdet3d_plugin/models/motion/decoder.py \
  scripts/multiseed campaigns/multiseed_v1 tests/manifests \
  analysis/multiseed_campaign.py
git commit -m "Install HyDrive closed-loop replication"

git status --porcelain
git -C Fail2Drive status --porcelain
```

Both status commands must produce no output. The generated campaign lock will
then record the new reconstruction commit and the patched Fail2Drive commit.

## 8. Continue with the infrastructure pilot

Follow `RUN_CAMPAIGN.md`. Do not start a CARLA server manually; each scheduled
job launches and terminates its own server.

## External sources

- SparseDriveV2 `bench2drive` quick start:
  <https://github.com/swc-17/SparseDriveV2/blob/bench2drive/docs/quick_start.md>
- Bench2Drive evaluator layout and CARLA notes:
  <https://github.com/Thinklab-SJTU/Bench2Drive>
- Fail2Drive plugin installation:
  <https://github.com/autonomousvision/fail2drive/tree/plugin>
