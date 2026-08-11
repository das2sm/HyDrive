# Derived from the SparseDriveV2/Bench2Drive evaluation agent and modified by
# HyDrive to add occupancy filtering, reselection, campaign logging, and strict
# evaluation controls. SparseDriveV2-derived portions are licensed under
# Apache-2.0; see licenses/SPARSEDRIVEV2_LICENSE and THIRD_PARTY_NOTICES.md.

import os
import json
import pathlib
import hashlib
import copy
import math
import random
import platform
import subprocess
import sys
import time
from pyquaternion import Quaternion
import cv2
import numpy as np
import torch
import carla

from team_code.pid_controller import PIDController
from team_code.planner import RoutePlanner

from team_code.guardian import Guardian  # GUARDIAN IMPORT
from team_code.route_logger import RouteLogger 
from team_code.collision_sensor import CollisionSensor
from team_code.pco import per_mode_occupancy_cost, select_pco_candidate

from leaderboard.autoagents import autonomous_agent
from leaderboard.utils.route_manipulation import _get_latlon_ref
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from mmcv import Config
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel
from mmcv.parallel.collate import collate as mm_collate_to_batch_form
from mmdet.models import build_detector
from mmdet.datasets.pipelines import Compose

from tools.visualization.visualize import Visualizer

IS_BENCH2DRIVE = os.environ.get('IS_BENCH2DRIVE', None)
CAMERAS = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']

'''
“The agent executes at CARLA’s 20 Hz simulation rate (fixed_delta_seconds=0.05). The inherited
frame_rate=10 variable affects SparseDrive temporal embeddings but not logging cadence or physical
timestep spacing.”
'''
frame_rate = 10
resize_scale = 0.44
save_interval = 200

lefthand_ego_to_lidar = np.array([[ 0, 1, 0, 0],
                                  [ 1, 0, 0, 0],
                                  [ 0, 0, 1, 0],
                                  [ 0, 0, 0, 1]])
left2right = np.eye(4)
left2right[1,1] = -1


def _git_commit():
    """Return the current git commit hash, or None if not in a repo."""
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            cwd=str(pathlib.Path(__file__).resolve().parent),
            text=True,
        ).strip()
    except Exception:
        return None


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as infile:
        for chunk in iter(lambda: infile.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def get_entry_point():
    return 'SparseDriveAgent'


class SparseDriveAgent(autonomous_agent.AutonomousAgent):
    def setup(self, path_to_conf_file):
        self.track = autonomous_agent.Track.SENSORS
        self.steer_step = 0
        
        config_parts = path_to_conf_file.split('+')
        self.config_path = config_parts[0]
        self.ckpt_path = config_parts[1]
        self.save_name = config_parts[2] if len(config_parts) >= 3 else "default_eval"
        self.gpu_rank = int(config_parts[3]) if len(config_parts) >= 4 else 0
        
        self.strict_campaign = os.environ.get(
            'HYDRIVE_STRICT_CAMPAIGN', '0'
        ) == '1'
        self.campaign_log_profile = os.environ.get(
            'HYDRIVE_LOG_PROFILE',
            'compact_campaign' if self.strict_campaign else 'full',
        )
        self.disable_visualizer = os.environ.get(
            'HYDRIVE_DISABLE_VISUALIZER',
            '1' if self.strict_campaign else '0',
        ) == '1'
        self.early_truncation_enabled = os.environ.get(
            'HYDRIVE_EARLY_TRUNCATION',
            '0' if self.strict_campaign else '1',
        ) == '1'
        self.diagnostic_collision_sensor_enabled = os.environ.get(
            'HYDRIVE_DIAGNOSTIC_COLLISION_SENSOR',
            '0' if self.strict_campaign else '1',
        ) == '1'

        campaign_fields = (
            'HYDRIVE_CAMPAIGN_ID',
            'HYDRIVE_CAMPAIGN_LOCK_SHA256',
            'HYDRIVE_SCHEDULE_SHA256',
            'HYDRIVE_JOB_ID',
            'HYDRIVE_JOB_DIR',
            'HYDRIVE_ARM',
            'HYDRIVE_ATTEMPT',
            'HYDRIVE_EVAL_SEED',
            'HYDRIVE_ORDER_POSITION',
            'TRAFFIC_MANAGER_SEED',
            'AGENT_SEED',
            'HYDRIVE_CONFIG_SHA256',
            'HYDRIVE_CHECKPOINT_SHA256',
            'HYDRIVE_FIXED_DELTA_SECONDS',
        )
        if self.strict_campaign:
            missing = [name for name in campaign_fields if name not in os.environ]
            if missing:
                raise RuntimeError(
                    f"Strict campaign environment is incomplete: {missing}"
                )
            if self.campaign_log_profile != 'compact_campaign':
                raise RuntimeError(
                    "Strict campaign requires HYDRIVE_LOG_PROFILE=compact_campaign"
                )

        self.seed = int(os.environ.get('AGENT_SEED', '42'))
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        
        self.step = -1
        self.initialized = False
        
        cfg = Config.fromfile(self.config_path)
        self.cfg = cfg
        
        self.pidcontroller = PIDController(cfg.get("pid_config"))
        
        if cfg.get("save_interval") is not None:
            self.save_interval = cfg.get("save_interval")
        else:
            self.save_interval = save_interval
        
        if hasattr(cfg, "plugin"):
            if cfg.plugin:
                import importlib
                if hasattr(cfg, "plugin_dir"):
                    plugin_dir = cfg.plugin_dir
                    _module_dir = os.path.dirname(plugin_dir)
                    _module_dir = _module_dir.split("/")
                    _module_path = _module_dir[0]
                    for m in _module_dir[1:]:
                        _module_path = _module_path + "." + m
                    print(_module_path)
                    plg_lib = importlib.import_module(_module_path)
                else:
                    _module_dir = os.path.dirname(self.config_path)
                    _module_dir = _module_dir.split("/")
                    _module_path = _module_dir[0]
                    for m in _module_dir[1:]:
                        _module_path = _module_path + "." + m
                    print(_module_path)
                    plg_lib = importlib.import_module(_module_path)
  
        model = build_detector(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
        checkpoint = load_checkpoint(model, self.ckpt_path, map_location='cpu', strict=True)
        
        self.model = MMDataParallel(model, device_ids=[self.gpu_rank])
        self.device = next(self.model.module.parameters()).device
        self.model.eval()
        
        self.test_pipeline = []
        for test_pipeline in cfg.test_pipeline:
            if test_pipeline["type"] not in ['LoadMultiViewImageFromFilesInCeph','LoadMultiViewImageFromFiles',]:
                self.test_pipeline.append(test_pipeline)
        self.test_pipeline = Compose(self.test_pipeline)
        
        self.data_aug_conf = cfg.data_aug_conf
        
        campaign_job_dir = os.environ.get('HYDRIVE_JOB_DIR')
        self.save_path = pathlib.Path(
            campaign_job_dir
            if campaign_job_dir
            else f'close_loop_log/save/{self.save_name}'
        )
        self.save_path.mkdir(parents=True, exist_ok=True)
        (self.save_path / 'meta').mkdir(exist_ok=True)
        
        plot_choices = dict(
            bev_pred = False,
            bev_gt = False,
            cam_pred = True,
            cam_gt = False,
            bevcam_pred = True,
            det = True,
            track = False,
            motion = True,
            map = True,
            planning = True,
            path = False,
            target_point = False,
            route = False,
            speed = True,
            det_attn_weight = True,
            map_attn_weight = True,
        )
        
        self.visualizer = None
        if not self.disable_visualizer:
            self.visualizer = Visualizer(
                plot_choices,
                self.save_path,
                planning_key=cfg.get("anchor_reference_group", "spatial"),
            )
        
        # ========== PCO RESCORING CONFIG ==========
        self.pco_rescore_enabled = bool(int(os.environ.get('PCO_RESCORE_ENABLED', '0')))
        self.pco_rescore_lambda = float(os.environ.get('PCO_RESCORE_LAMBDA', '0.0'))
        self.pco_rescore_topk = int(os.environ.get('PCO_RESCORE_TOPK', '0'))
        self.pco_rescore_fallback = os.environ.get('PCO_RESCORE_FALLBACK', 'keep_original')
        self.pco_rescore_score_source = os.environ.get('PCO_RESCORE_SCORE_SOURCE', 'post_rescore')
        self.pco_rescore_cost_mode = os.environ.get('PCO_RESCORE_COST_MODE', 'binary')
        self.pco_rescore_log_full_costs = bool(int(os.environ.get('PCO_RESCORE_LOG_FULL_COSTS', '0')))
        self.pco_occupancy_source = os.environ.get('PCO_OCCUPANCY_SOURCE', 'current_frame')
        self.pco_selection_policy = os.environ.get('PCO_SELECTION_POLICY', 'binary_veto')
        self.pco_progress_tolerance_m = float(os.environ.get('PCO_PROGRESS_TOLERANCE_M', '2.0'))
        _valid_fallbacks = {'keep_original'}
        assert self.pco_rescore_fallback in _valid_fallbacks, (
            f"PCO_RESCORE_FALLBACK={self.pco_rescore_fallback!r} "
            f"not in {_valid_fallbacks}"
        )
        _valid_score_sources = {'pre_rescore', 'post_rescore'}
        assert self.pco_rescore_score_source in _valid_score_sources, (
            f"PCO_RESCORE_SCORE_SOURCE={self.pco_rescore_score_source!r} "
            f"not in {_valid_score_sources}"
        )
        _valid_occupancy_sources = {'current_frame', 'temporal_cv'}
        assert self.pco_occupancy_source in _valid_occupancy_sources, (
            f"PCO_OCCUPANCY_SOURCE={self.pco_occupancy_source!r} "
            f"not in {_valid_occupancy_sources}"
        )
        _valid_selection_policies = {'binary_veto', 'progress_preserving_veto'}
        assert self.pco_selection_policy in _valid_selection_policies, (
            f"PCO_SELECTION_POLICY={self.pco_selection_policy!r} "
            f"not in {_valid_selection_policies}"
        )
        # Waypoint horizon times for temporal occupancy (matches planner waypoint spacing)
        self.pco_temporal_waypoint_times = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        # ========== PCO HYSTERESIS ==========
        # Trajectory-space hysteresis margin η (meters).
        # When hysteresis-retained candidate's traj-distance to original
        # is within η of the V2-selected candidate's distance, prefer
        # the retained candidate (closest to previous trajectory).
        self.pco_hysteresis_margin = float(os.environ.get('PCO_HYSTERESIS_MARGIN', '0.5'))
        # ===================================

        if self.strict_campaign:
            arm = os.environ['HYDRIVE_ARM']
            expected = {
                'baseline': (False, 'current_frame'),
                'current_frame': (True, 'current_frame'),
                'temporal': (True, 'temporal_cv'),
            }
            if arm not in expected:
                raise RuntimeError(f"Unknown strict campaign arm: {arm!r}")
            expected_enabled, expected_source = expected[arm]
            invariants = {
                'PCO_RESCORE_ENABLED': self.pco_rescore_enabled == expected_enabled,
                'PCO_OCCUPANCY_SOURCE': self.pco_occupancy_source == expected_source,
                'PCO_RESCORE_COST_MODE': self.pco_rescore_cost_mode == 'binary',
                'PCO_SELECTION_POLICY': self.pco_selection_policy == 'binary_veto',
                'PCO_RESCORE_TOPK': self.pco_rescore_topk == 0,
                'PCO_RESCORE_SCORE_SOURCE': self.pco_rescore_score_source == 'post_rescore',
                'PCO_RESCORE_FALLBACK': self.pco_rescore_fallback == 'keep_original',
                'PCO_HYSTERESIS_MARGIN': self.pco_hysteresis_margin == 0.0,
                'PCO_RESCORE_LOG_FULL_COSTS': not self.pco_rescore_log_full_costs,
            }
            failures = [name for name, valid in invariants.items() if not valid]
            if failures:
                raise RuntimeError(
                    f"Strict campaign arm {arm!r} violates frozen settings: {failures}"
                )

        if self.pco_rescore_enabled:
            print(f"[PCO Rescore] Enabled, λ={self.pco_rescore_lambda}, "
                  f"topk={self.pco_rescore_topk}, cost_mode={self.pco_rescore_cost_mode}, "
                  f"fallback={self.pco_rescore_fallback}, score_source={self.pco_rescore_score_source}, "
                  f"occupancy_source={self.pco_occupancy_source}, "
                  f"selection_policy={self.pco_selection_policy}, "
                  f"progress_tolerance_m={self.pco_progress_tolerance_m}, "
                  f"hysteresis_margin={self.pco_hysteresis_margin}")

        # ========== GUARDIAN INITIALIZATION ==========
        self.use_guardian = True  # Toggle to enable/disable Guardian
        if self.use_guardian:
            self.guardian = Guardian(
                world=None,  # Will be set in _init()
                debug=False,  # Set True to visualize in CARLA
                exclude_static_bbs=('Poles',),
            )
            print("[SparseDrive] Guardian initialized")
        else:
            self.guardian = None
        # =============================================

        # ========== ROUTE LOGGER ==========
        strict_metadata = {
            'git_commit': _git_commit(),
            'config_path': str(pathlib.Path(self.config_path).resolve()),
            'checkpoint_path': str(pathlib.Path(self.ckpt_path).resolve()),
            'config_sha256': (
                os.environ['HYDRIVE_CONFIG_SHA256']
                if self.strict_campaign else _sha256_file(self.config_path)
            ),
            'checkpoint_sha256': (
                os.environ['HYDRIVE_CHECKPOINT_SHA256']
                if self.strict_campaign else _sha256_file(self.ckpt_path)
            ),
            'python_version': sys.version,
            'platform': platform.platform(),
            'numpy_version': np.__version__,
            'torch_version': torch.__version__,
            'torch_cuda_version': torch.version.cuda,
            'cudnn_version': torch.backends.cudnn.version(),
            'gpu_name': (
                torch.cuda.get_device_name(self.gpu_rank)
                if torch.cuda.is_available() else None
            ),
            'carla_version': getattr(carla, '__version__', None),
            'agent_seed': self.seed,
            'traffic_manager_seed': int(
                os.environ.get('TRAFFIC_MANAGER_SEED', '0')
            ),
            'campaign_id': os.environ.get('HYDRIVE_CAMPAIGN_ID'),
            'campaign_lock_sha256': os.environ.get(
                'HYDRIVE_CAMPAIGN_LOCK_SHA256'
            ),
            'schedule_sha256': os.environ.get('HYDRIVE_SCHEDULE_SHA256'),
            'job_id': os.environ.get('HYDRIVE_JOB_ID'),
            'arm': os.environ.get('HYDRIVE_ARM'),
            'attempt': int(os.environ.get('HYDRIVE_ATTEMPT', '0')),
            'evaluation_seed': int(os.environ.get('HYDRIVE_EVAL_SEED', '0')),
            'order_position': int(os.environ.get('HYDRIVE_ORDER_POSITION', '0')),
            'occupancy_source': self.pco_occupancy_source,
            'filter_enabled': self.pco_rescore_enabled,
            'selection_policy': 'binary_veto',
            'waypoint_times_seconds': self.pco_temporal_waypoint_times,
        }
        self.route_logger = RouteLogger(
            log_dir=str(self.save_path / 'route_logs'),
            horizon_seconds=3.0,
            fps=20,
            planner_waypoint_times_seconds=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
            occupancy_cell_size_m=0.5,
            log_profile=self.campaign_log_profile,
            run_metadata=(strict_metadata if self.strict_campaign else {
                'git_commit': _git_commit(),
                'config_path': str(pathlib.Path(self.config_path).resolve()),
                'checkpoint_path': str(pathlib.Path(self.ckpt_path).resolve()),
                'config_sha256': _sha256_file(self.config_path),
                'checkpoint_sha256': _sha256_file(self.ckpt_path),
                'python_version': sys.version,
                'platform': platform.platform(),
                'numpy_version': np.__version__,
                'torch_version': torch.__version__,
                'torch_cuda_version': torch.version.cuda,
                'cudnn_version': torch.backends.cudnn.version(),
                'gpu_name': (
                    torch.cuda.get_device_name(self.gpu_rank)
                    if torch.cuda.is_available() else None
                ),
                'carla_version': getattr(carla, '__version__', None),
                'seed': self.seed,
                'traffic_manager_seed': int(
                    os.environ.get('TRAFFIC_MANAGER_SEED', '0')
                ),
                'guardian_longitudinal_margin_m': self.guardian.longitudinal_margin,
                'guardian_lateral_margin_m': self.guardian.lateral_margin,
                'guardian_sweep_sample_spacing_m': self.guardian.sweep_sample_spacing,
                'guardian_expensive_step_interval': (
                    self.guardian.expensive_step_interval
                ),
                'guardian_actor_cache_max_age': (
                    self.guardian._actor_list_cache_max_age
                ),
                'guardian_excluded_static_bbs': sorted(
                    self.guardian.exclude_static_bbs
                ),
                'pco_rescore_enabled': self.pco_rescore_enabled,
                'pco_rescore_lambda': self.pco_rescore_lambda,
                'pco_rescore_topk': self.pco_rescore_topk,
                'pco_rescore_fallback': self.pco_rescore_fallback,
                'pco_rescore_score_source': self.pco_rescore_score_source,
                'pco_rescore_cost_mode': self.pco_rescore_cost_mode,
                'pco_rescore_log_full_costs': self.pco_rescore_log_full_costs,
                'pco_occupancy_source': self.pco_occupancy_source,
                'pco_selection_policy': self.pco_selection_policy,
                'pco_progress_tolerance_m': self.pco_progress_tolerance_m,
                'pco_hysteresis_margin': self.pco_hysteresis_margin,
                'pco_temporal_waypoint_times': self.pco_temporal_waypoint_times,
                'strict_campaign': self.strict_campaign,
                'campaign_id': os.environ.get('HYDRIVE_CAMPAIGN_ID'),
                'campaign_lock_sha256': os.environ.get(
                    'HYDRIVE_CAMPAIGN_LOCK_SHA256'
                ),
                'schedule_sha256': os.environ.get('HYDRIVE_SCHEDULE_SHA256'),
                'job_id': os.environ.get('HYDRIVE_JOB_ID'),
                'arm': os.environ.get('HYDRIVE_ARM'),
                'attempt': int(os.environ.get('HYDRIVE_ATTEMPT', '0')),
                'evaluation_seed': int(os.environ.get('HYDRIVE_EVAL_SEED', '0')),
                'order_position': int(
                    os.environ.get('HYDRIVE_ORDER_POSITION', '0')
                ),
                'log_profile': self.campaign_log_profile,
                'early_truncation_enabled': self.early_truncation_enabled,
                'diagnostic_collision_sensor_enabled': (
                    self.diagnostic_collision_sensor_enabled
                ),
            }),
        )
        print("[SparseDrive] Route logger initialized")
        # =================================

        self.collision_sensor = None  # Will be initialized in _init() when we have access to the world and ego actor
        self.collision_latched = False
        self._stuck_counter = None

        # Trajectory-space hysteresis state
        self._hysteresis_traj_world = None  # (T, 2) previous trajectory in world XY
        self._hysteresis_retained_count = 0  # frames this retained trajectory has persisted
   
        self.lidar2cam = {
        'CAM_FRONT':np.array([[ 1.  ,  0.  ,  0.  ,  0.  ],
                                [ 0.  ,  0.  ,  1.  ,  0.  ],
                                [ 0.  , -1.  ,  0.  ,  0.  ],
                                [ 0.  , -0.24, -1.19,  1.  ]]),
        'CAM_FRONT_RIGHT':np.array([[ 0.57357644,  0.        ,  0.81915204,  0.        ],
                                    [-0.81915204,  0.        ,  0.57357644,  0.        ],
                                    [ 0.        , -1.        ,  0.        ,  0.        ],
                                    [ 0.22517331, -0.24      , -0.82909407,  1.        ]]),
        'CAM_FRONT_LEFT':np.array([[ 0.57357644,  0.        , -0.81915204,  0.        ],
                                    [ 0.81915204,  0.        ,  0.57357644,  0.        ],
                                    [ 0.        , -1.        ,  0.        ,  0.        ],
                                    [-0.22517331, -0.24      , -0.82909407,  1.        ]]),
        'CAM_BACK':np.array([[-1.00000000e+00,  0.00000000e+00,  1.22464680e-16, 0.00000000e+00],
                            [-1.22464680e-16,  0.00000000e+00, -1.00000000e+00, 0.00000000e+00],
                            [ 0.00000000e+00, -1.00000000e+00,  0.00000000e+00, 0.00000000e+00],
                            [-1.97168135e-16, -2.40000000e-01, -1.61000000e+00, 1.00000000e+00]]),
        'CAM_BACK_LEFT':np.array([[-0.34202014,  0.        , -0.93969262,  0.        ],
                                    [ 0.93969262,  0.        , -0.34202014,  0.        ],
                                    [ 0.        , -1.        ,  0.        ,  0.        ],
                                    [-0.25388956, -0.24      , -0.49288953,  1.        ]]),
        'CAM_BACK_RIGHT':np.array([[-0.34202014,  0.        ,  0.93969262,  0.        ],
                                    [-0.93969262,  0.        , -0.34202014,  0.        ],
                                    [ 0.        , -1.        ,  0.        ,  0.        ],
                                    [ 0.25388956, -0.24      , -0.49288953,  1.        ]])
        }
        
        self.cam_intrinsic = {
        'CAM_FRONT': np.array([[1.14251841e+03, 0.00000000e+00, 8.00000000e+02],
                            [0.00000000e+00, 1.14251841e+03, 4.50000000e+02],
                            [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),
        'CAM_FRONT_RIGHT': np.array([[1.14251841e+03, 0.00000000e+00, 8.00000000e+02],
                                    [0.00000000e+00, 1.14251841e+03, 4.50000000e+02],
                                    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),
        'CAM_FRONT_LEFT': np.array([[1.14251841e+03, 0.00000000e+00, 8.00000000e+02],
                                    [0.00000000e+00, 1.14251841e+03, 4.50000000e+02],
                                    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),
        'CAM_BACK':np.array([[560.16603057,   0.        , 800.        ],
                            [  0.        , 560.16603057, 450.        ],
                            [  0.        ,   0.        ,   1.        ]]),
        'CAM_BACK_LEFT':np.array([[1.14251841e+03, 0.00000000e+00, 8.00000000e+02],
                                    [0.00000000e+00, 1.14251841e+03, 4.50000000e+02],
                                    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),
        'CAM_BACK_RIGHT':np.array([[1.14251841e+03, 0.00000000e+00, 8.00000000e+02],
                                    [0.00000000e+00, 1.14251841e+03, 4.50000000e+02],
                                    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),
        }
        
        self.lidar2img = {}
        for key, value in self.cam_intrinsic.items():
            transform_matrix = np.eye(3)
            transform_matrix[:2, :2] *= resize_scale
            intrinsic = transform_matrix @ value
            self.cam_intrinsic[key] = intrinsic
            viewpad = np.eye(4)
            viewpad[: intrinsic.shape[0], : intrinsic.shape[1]] = intrinsic
            lidar2cam = self.lidar2cam[key]
            self.lidar2img[key] = viewpad @ lidar2cam.T
        
        self.lidar2ego = np.array([[ 0. ,  1. ,  0. , -0.39],
                                   [-1. ,  0. ,  0. ,  0.  ],
                                   [ 0. ,  0. ,  1. ,  1.84],
                                   [ 0. ,  0. ,  0. ,  1.  ]])
        
        self.bev_extrinsic = np.array([[1.0,  0.0,  0.0,  0.0], 
                                      [0.0, -1.0,  0.0,  0.0], 
                                      [0.0,  0.0, -1.0, 50.0], 
                                      [0.0,  0.0,  0.0,  1.0]])
        self.bev_intrinsic = np.array([[548.993771650447, 0.0, 256.0, 0], [0.0, 548.993771650447, 256.0, 0], [0.0, 0.0, 1.0, 0], [0, 0, 0, 1.0]])
        self.bev2img = self.bev_intrinsic @ self.bev_extrinsic
        

    
    def _init(self):
        if self.strict_campaign:
            settings = CarlaDataProvider.get_world().get_settings()
            expected_delta = float(os.environ['HYDRIVE_FIXED_DELTA_SECONDS'])
            if not settings.synchronous_mode:
                raise RuntimeError("Strict campaign requires synchronous CARLA mode")
            if (
                settings.fixed_delta_seconds is None
                or not math.isclose(
                    float(settings.fixed_delta_seconds), expected_delta,
                    rel_tol=0.0, abs_tol=1e-9,
                )
            ):
                raise RuntimeError(
                    "Strict campaign CARLA timestep mismatch: "
                    f"{settings.fixed_delta_seconds} != {expected_delta}"
                )
            self.route_logger.run_metadata.update({
                'synchronous_mode': True,
                'fixed_delta_seconds': expected_delta,
            })
        self.lat_ref, self.lon_ref = _get_latlon_ref(CarlaDataProvider.get_world()) 
        
        self._route_planner = RoutePlanner(4.0, 50.0, lat_ref=self.lat_ref, lon_ref=self.lon_ref)
        self._route_planner.set_route(self._global_plan, True)
        
        global_plan_far = getattr(self, '_global_plan_far', self._global_plan)
        self._route_planner_far = RoutePlanner(4.0, 50.0, lat_ref=self.lat_ref, lon_ref=self.lon_ref)
        self._route_planner_far.set_route(global_plan_far, True)
        
        # ========== SET GUARDIAN WORLD ==========
        if self.guardian is not None:
            self.guardian.world = CarlaDataProvider.get_world()
            print("[Guardian] World reference set")
        # ========================================
        
        self.initialized = True
        self.metric_info = {}
    
    def sensors(self):
        W = 1600 * resize_scale
        H = 900 * resize_scale
        
        sensors =[
                # camera rgb
                {
                    'type': 'sensor.camera.rgb',
                    'x': 0.80, 'y': 0.0, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'width': W, 'height': H, 'fov': 70,
                    'id': 'CAM_FRONT'
                },
                {
                    'type': 'sensor.camera.rgb',
                    'x': 0.27, 'y': -0.55, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': -55.0,
                    'width': W, 'height': H, 'fov': 70,
                    'id': 'CAM_FRONT_LEFT'
                },
                {
                    'type': 'sensor.camera.rgb',
                    'x': 0.27, 'y': 0.55, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 55.0,
                    'width': W, 'height': H, 'fov': 70,
                    'id': 'CAM_FRONT_RIGHT'
                },
                {
                    'type': 'sensor.camera.rgb',
                    'x': -2.0, 'y': 0.0, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 180.0,
                    'width': W, 'height': H, 'fov': 110,
                    'id': 'CAM_BACK'
                },
                {
                    'type': 'sensor.camera.rgb',
                    'x': -0.32, 'y': -0.55, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': -110.0,
                    'width': W, 'height': H, 'fov': 70,
                    'id': 'CAM_BACK_LEFT'
                },
                {
                    'type': 'sensor.camera.rgb',
                    'x': -0.32, 'y': 0.55, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 110.0,
                    'width': W, 'height': H, 'fov': 70,
                    'id': 'CAM_BACK_RIGHT'
                },
                # imu
                {
                    'type': 'sensor.other.imu',
                    'x': -1.4, 'y': 0.0, 'z': 0.0,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'sensor_tick': 0.05,
                    'id': 'IMU'
                },
                # gps
                {
                    'type': 'sensor.other.gnss',
                    'x': -1.4, 'y': 0.0, 'z': 0.0,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'sensor_tick': 0.01,
                    'id': 'GPS'
                },
                # speed
                {
                    'type': 'sensor.speedometer',
                    'reading_frequency': 20,
                    'id': 'SPEED'
                },
                # lidar
                {   'type': 'sensor.lidar.ray_cast',
                    'x': -0.39, 'y': 0.0, 'z': 1.84,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'range': 85,
                    'rotation_frequency': 10,
                    'channels': 64,
                    'points_per_second': 600000,
                    'dropoff_general_rate': 0.0,
                    'dropoff_intensity_limit': 0.0,
                    'dropoff_zero_intensity': 0.0,
                    'id': 'LIDAR_TOP'
                },
            ]
        
        if IS_BENCH2DRIVE:
            sensors += [
                    {	
                        'type': 'sensor.camera.rgb',
                        'x': 0.0, 'y': 0.0, 'z': 50.0,
                        'roll': 0.0, 'pitch': -90.0, 'yaw': 0.0,
                        'width': 512, 'height': 512, 'fov': 5 * 10.0,
                        'id': 'bev'
                    }]
        
        return sensors
    
    def tick(self, input_data):
        self.step += 1
        
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 20]
        imgs = {}
        for cam in CAMERAS:
            img = input_data[cam][1][:, :, :3]
            _, img = cv2.imencode('.jpg', img, encode_param)
            img = cv2.imdecode(img, cv2.IMREAD_COLOR)
            imgs[cam] = img
        
        bev = cv2.cvtColor(input_data['bev'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        gps = input_data['GPS'][1][:2]
        speed = input_data['SPEED'][1]['speed']
        compass = input_data['IMU'][1][-1]
        acceleration = input_data['IMU'][1][:3]
        angular_velocity = input_data['IMU'][1][3:6]
  
        lidar = CarlaDataProvider.get_world().get_actors().filter('*sensor.lidar.ray_cast*')[0]
        world2lidar = lidar.get_transform().get_inverse_matrix()
        world2lidar = lefthand_ego_to_lidar @ world2lidar @ left2right
        
        lidar2global =  self.invert_pose(world2lidar)
        ego2global = self.invert_pose(world2lidar) @ self.invert_pose(self.lidar2ego)
        global2ego = self.lidar2ego @ world2lidar
        
        pos = np.copy(ego2global[0:2, 3])
        pos[1] *= -1
        
        near_node, near_command = self._route_planner.run_step(pos)
        far_node, far_command = self._route_planner_far.run_step(pos)
        
        if (math.isnan(compass) == True):
            compass = 0.0
            acceleration = np.zeros(3)
            angular_velocity = np.zeros(3)
        
        result = {
            'imgs': imgs,
            'gps': gps,
            'pos':pos,
            'speed': speed,
            'compass': compass,
            'bev': bev,
            'acceleration':acceleration,
            'angular_velocity':angular_velocity,
            'command_near':near_command,
            'command_near_xy':near_node,
            'command_far':far_command,
            'command_far_xy':far_node,
            'lidar2global': lidar2global,
            'global2lidar': world2lidar,
            'ego2global': ego2global,
            'global2ego': global2ego,
        }

        return result
    
    @torch.no_grad()
    def _local_to_world_xy(self, points_forward_right, ego_transform):
        """Convert CARLA-local [forward, right] to world XY."""
        yaw = np.deg2rad(ego_transform.rotation.yaw)
        c, s = np.cos(yaw), np.sin(yaw)
        x = ego_transform.location.x + points_forward_right[:, 0] * c - points_forward_right[:, 1] * s
        y = ego_transform.location.y + points_forward_right[:, 0] * s + points_forward_right[:, 1] * c
        return np.stack([x, y], axis=1)

    def _world_xy_to_carla_local(self, points_world_xy, ego_transform):
        """Convert world XY to CARLA-local [forward, right]."""
        yaw = np.deg2rad(ego_transform.rotation.yaw)
        c, s = np.cos(yaw), np.sin(yaw)
        dx = points_world_xy[:, 0] - ego_transform.location.x
        dy = points_world_xy[:, 1] - ego_transform.location.y
        forward = dx * c + dy * s
        right = -dx * s + dy * c
        return np.stack([forward, right], axis=1)

    def _planner_to_carla_local(self, traj_planner):
        """Convert planner [right, forward] to CARLA [forward, right]."""
        return traj_planner[:, [1, 0]]

    def _carla_local_to_planner(self, traj_carla_local):
        """Convert CARLA [forward, right] to planner [right, forward]."""
        return traj_carla_local[:, [1, 0]]

    def run_step(self, input_data, timestamp):
        if not self.initialized:
            self._init()
        
        tick_data = self.tick(input_data)
        
        results = {}
        results['timestamp'] = self.step / frame_rate
        results['img'] = []
        results['lidar2img'] = []
        results['lidar2cam'] = []
        results['cam_intrinsic'] = []
        
        for cam in CAMERAS:
            results['img'].append(tick_data['imgs'][cam])
            results['lidar2img'].append(self.lidar2img[cam])
            results['lidar2cam'].append(self.lidar2cam[cam])
            results['cam_intrinsic'].append(copy.deepcopy(self.cam_intrinsic[cam]))
        
        results["bev_img"] = tick_data["bev"]        
        results["bev2img"] = self.bev2img
        results["bev_extrinsic"] = self.bev_extrinsic
        results["bev_intrinsic"] = self.bev_intrinsic
        results["lidar2global"] = tick_data["lidar2global"]
        
        raw_theta = tick_data['compass']   if not np.isnan(tick_data['compass']) else 0
        ego_theta = -raw_theta + np.pi/2
        rotation = list(Quaternion(axis=[0, 0, 1], radians=ego_theta))
        
        can_bus = np.zeros(18)
        can_bus[0] = tick_data['pos'][0]
        can_bus[1] = -tick_data['pos'][1]
        can_bus[3:7] = rotation
        can_bus[7] = tick_data['speed']
        can_bus[10:13] = tick_data['acceleration']
        can_bus[11] *= -1
        can_bus[13:16] = -tick_data['angular_velocity']
        can_bus[16] = ego_theta
        can_bus[17] = ego_theta / np.pi * 180 
        results['can_bus'] = can_bus
        
        ego_status = np.zeros(10, dtype=np.float32)
        ego_status[:3] = np.array([tick_data['acceleration'][0],-tick_data['acceleration'][1],tick_data['acceleration'][2]])
        ego_status[3:6] = -np.array(tick_data['angular_velocity'])
        ego_status[6:9] = np.array([tick_data['speed'],0,0])
        results["ego_status"] = ego_status
        
        command = tick_data['command_near']
        if command < 0:
            command = 4
        command -= 1
        command_onehot = np.zeros(6)
        command_onehot[command] = 1
        results['gt_ego_fut_cmd'] = command_onehot
        
        theta_to_lidar = raw_theta
        
        def global2ego(point):
            ego2global = tick_data["ego2global"]
            ego_trans = ego2global[:3,3]
            point = np.array([point[0]-ego_trans[0],-point[1]-ego_trans[1]])
            rotation_matrix = np.array([[np.cos(theta_to_lidar),-np.sin(theta_to_lidar)],[np.sin(theta_to_lidar),np.cos(theta_to_lidar)]])
            local_point = rotation_matrix @ point
            return local_point
        
        local_command_xy = global2ego(tick_data['command_near_xy'])
        results['tp_near'] = local_command_xy
        
        local_command_xy_far = global2ego(tick_data['command_far_xy'])
        results['tp_far'] = local_command_xy_far
        
        local_route = []
        for (global_point, _) in self._route_planner.route:
            local_point = global2ego(global_point)
            local_route.append(local_point)
        results["route"] = np.array(local_route)
        
        stacked_img = np.stack(results['img'], axis=-1)
        results['img_shape'] = stacked_img.shape
        results['ori_shape'] = stacked_img.shape
        results['pad_shape'] = stacked_img.shape
        
        aug_config = self.get_augmentation()
        results["aug_config"] = aug_config
        
        self.data = copy.deepcopy(results)
        
        results = self.test_pipeline(results)
        
        if "route" in results:
            self.data["route"] = results["route"]
        
        input_data_batch = mm_collate_to_batch_form([results], samples_per_gpu=1)
        for key, data in input_data_batch.items():
            if key != 'img_metas':
                if torch.is_tensor(data):
                    data = data.to(self.device)
        
        output_data_batch = self.model(**input_data_batch)
        
        output = output_data_batch[0]['img_bbox']
         
        # ========== EXTRACT SPARSEDRIVE TRAJECTORY ==========
        # SparseDrive outputs trajectory in 'traj_final'
        sparsedrive_traj = None
        if 'traj_final' in output:
            sparsedrive_traj = output['traj_final']
            
            if torch.is_tensor(sparsedrive_traj):
                sparsedrive_traj = sparsedrive_traj.cpu().numpy()
    
        else:
            raise RuntimeError(
                f"'traj_final' not found in model output keys: {output.keys()}"
            )

        # ========== EXTRACT EGO MULTI-MODAL PLANNER DISTRIBUTION ==========
        planner_trajs = None
        planner_scores = None
        planner_logits = None
        planner_logits_masked = None
        planner_scores_masked = None
        planner_rescore_mask = None
        planner_collision_rescore_mask = None
        planner_detection_rescore_mask = None
        planner_selected_mode_index = None

        if 'traj_reg' not in output or 'traj_cls' not in output:
            raise RuntimeError(
                "Planner trajectory or scores not found in model output. "
                f"Keys: {output.keys()}"
            )

        else:   
            planner_trajs = output['traj_reg']          # (1024, 6, 2) 
            # Weighted PCO uses learned planner logits before any collision or
            # detection-based decoder rescoring. Rescore state is logged
            # separately so it cannot be mistaken for model uncertainty.
            required_planner_keys = [
                'traj_cls_logits_post_rescore',
                'traj_rescore_mask',
                'traj_selected_mode_index',
            ]
            if not self.strict_campaign:
                required_planner_keys.extend([
                    'traj_cls_logits_pre_rescore',
                    'traj_collision_rescore_mask',
                    'traj_detection_rescore_mask',
                ])
            missing_planner_keys = [
                key for key in required_planner_keys if key not in output
            ]
            assert not missing_planner_keys, (
                f"Missing planner provenance keys {missing_planner_keys}; "
                f"keys={list(output.keys())}"
            )
            planner_logits_masked = output['traj_cls_logits_post_rescore']
            planner_rescore_mask = output['traj_rescore_mask']
            planner_selected_mode_index = output['traj_selected_mode_index']
            if not self.strict_campaign:
                planner_logits = output['traj_cls_logits_pre_rescore']
                planner_collision_rescore_mask = output[
                    'traj_collision_rescore_mask'
                ]
                planner_detection_rescore_mask = output[
                    'traj_detection_rescore_mask'
                ]
                planner_scores = torch.softmax(planner_logits, dim=-1)
                planner_scores_masked = torch.softmax(
                    planner_logits_masked,
                    dim=-1,
                )
            planner_trajs = planner_trajs.detach().cpu().numpy()  # (1024, 6, 2)
            planner_logits_masked = planner_logits_masked.detach().cpu().numpy()
            planner_rescore_mask = planner_rescore_mask.detach().cpu().numpy()
            if not self.strict_campaign:
                planner_scores = planner_scores.detach().cpu().numpy()
                planner_logits = planner_logits.detach().cpu().numpy()
                planner_scores_masked = planner_scores_masked.detach().cpu().numpy()
                planner_collision_rescore_mask = (
                    planner_collision_rescore_mask.detach().cpu().numpy()
                )
                planner_detection_rescore_mask = (
                    planner_detection_rescore_mask.detach().cpu().numpy()
                )
            planner_selected_mode_index = int(planner_selected_mode_index)

            assert planner_trajs.shape == (1024, 6, 2), f"Wrong shape: {planner_trajs.shape}"

        # ========== GET EGO STATE FOR GUARDIAN ==========
        ego_actor = CarlaDataProvider.get_hero_actor()
        if (
            self.diagnostic_collision_sensor_enabled
            and self.collision_sensor is None
            and ego_actor is not None
        ):
            self.collision_sensor = CollisionSensor(ego_actor)
            print("[CollisionSensor] Attached to ego vehicle")
        ego_transform = ego_actor.get_transform()
        ego_speed = tick_data['speed']
        
        # ========== COMPUTE OCCUPANCY ==========
        guardian_intervene = False
        guardian_brake = 0.0

        occ_grid = np.zeros((120,120), dtype=np.float32)
        occ_meta = {'source': 'none', 'actor_count': 0, 'obstacle_count': 0}

        if self.guardian is not None:
            try:
                if self.strict_campaign:
                    if os.environ['HYDRIVE_ARM'] != 'baseline':
                        self.guardian._refresh_ego_geometry(ego_actor)
                else:
                    guardian_intervene, guardian_brake = self.guardian.evaluate(
                        traj=sparsedrive_traj,
                        ego_transform=ego_transform,
                        speed=ego_speed,
                        ego_actor=ego_actor
                    )
                    occ_grid = self.guardian.latest_occ_grid
                    occ_meta = self.guardian.latest_occ_meta

            except Exception as e:
                print(f"[Guardian ERROR] {e}")
                import traceback
                traceback.print_exc()
                occ_meta = {'source': 'error', 'actor_count': 0, 'obstacle_count': 0, 'skipped_actor_count': 0}
        
        # ========== DETECT COLLISION ==========
        collision_occurred = False
        collision_event = None

        if (
            self.collision_sensor is not None and
            self.collision_sensor.has_collision() and
            not self.collision_latched
        ):
            collision_occurred = True
            collision_event = self.collision_sensor.get_latest_collision()
            self.collision_latched = True

        try:
            simulator_snapshot = ego_actor.get_world().get_snapshot()
            simulator_frame = int(simulator_snapshot.frame)
            simulator_elapsed_seconds = float(
                simulator_snapshot.timestamp.elapsed_seconds
            )
        except Exception:
            simulator_frame = None
            simulator_elapsed_seconds = None

        # ========== PCO RESCORING ==========
        pco_selected_mode_original = planner_selected_mode_index
        pco_traj_original = sparsedrive_traj.copy()
        pco_fallback_reason = 'disabled'
        pco_mode_changed = False
        pco_cost_selected_original = np.nan
        pco_cost_selected_rescored = np.nan
        pco_cost_min = np.nan
        pco_cost_mean = np.nan
        pco_cost_max = np.nan
        pco_valid_mode_fraction = 0.0
        pco_valid_candidate_count = 0
        pco_clear_candidate_count = 0
        pco_safe_candidate_rank = -1
        pco_latency_ms = 0.0
        pco_selected_mode_rescored = planner_selected_mode_index
        pco_traj_rescored = sparsedrive_traj.copy()
        pco_full_costs = None
        pco_temporal_build_latency_ms = 0.0
        pco_temporal_dynamic_actor_count = 0
        pco_temporal_static_actor_count = 0
        pco_temporal_skipped_actor_count = 0
        pco_temporal_valid = False
        pco_occupancy_valid = False
        pco_selection_policy_logged = self.pco_selection_policy
        pco_progress_tolerance_m_logged = self.pco_progress_tolerance_m
        pco_progress_original = -1.0
        pco_progress_selected = -1.0
        pco_progress_max_safe = -1.0
        pco_progress_threshold = -1.0
        pco_progress_eligible_count = -1
        pco_progress_filter_active = False
        pco_traj_distance_selected = -1.0
        pco_hysteresis_active = False
        pco_hysteresis_retained_count = self._hysteresis_retained_count
        pco_planner_modes_lon = output['lon_reg'].shape[0] if 'lon_reg' in output else 6

        if self.pco_rescore_enabled and self.guardian is not None:
            t_start = time.perf_counter()

            # Build occupancy grid(s) — current-frame or temporal
            if (
                self.pco_occupancy_source == 'temporal_cv'
                or (self.strict_campaign and self.pco_occupancy_source == 'current_frame')
            ):
                t_build_start = time.perf_counter()
                try:
                    build_times = (
                        self.pco_temporal_waypoint_times
                        if self.pco_occupancy_source == 'temporal_cv'
                        else [0.0] * len(self.pco_temporal_waypoint_times)
                    )
                    temporal_grids, temporal_meta = self.guardian.build_temporal_occupancy_grids(
                        ego_transform,
                        build_times,
                    )
                    pco_temporal_valid = True
                    cost_occ_grid = temporal_grids  # (T, H, W)
                except Exception as exc:
                    print(f"[PCO] Temporal occupancy build failed: {exc}")
                    temporal_meta = {}
                    pco_temporal_valid = False
                    cost_occ_grid = None
                pco_temporal_build_latency_ms = (time.perf_counter() - t_build_start) * 1000.0
                pco_temporal_dynamic_actor_count = temporal_meta.get('dynamic_actor_count', 0)
                pco_temporal_static_actor_count = temporal_meta.get('static_actor_count', 0)
                pco_temporal_skipped_actor_count = temporal_meta.get('skipped_actor_count', 0)
            else:
                if occ_meta.get('source') == 'error':
                    cost_occ_grid = None
                else:
                    cost_occ_grid = occ_grid

            # Guard: invalid occupancy → skip veto
            if cost_occ_grid is None:
                pco_fallback_reason = 'invalid_costs'
                pco_latency_ms = (time.perf_counter() - t_start) * 1000.0
            else:
                # Sigmoid scores matching planner's actual selection basis.
                # Logged planner_scores remain softmax for backward compat.
                logits = (
                    planner_logits
                    if self.pco_rescore_score_source == 'pre_rescore'
                    else planner_logits_masked
                )
                sigmoid_scores = 1.0 / (1.0 + np.exp(-logits))

                cost_ok = True
                try:
                    costs, first_hits, valid_modes, diag = per_mode_occupancy_cost(
                        planner_trajs,
                        cost_occ_grid,
                        footprint_L=self.guardian.vehicle_length,
                        footprint_W=self.guardian.vehicle_width,
                        cost_mode=self.pco_rescore_cost_mode,
                    )
                except Exception as exc:
                    print(f"[PCO] Cost computation failed: {exc}")
                    cost_ok = False

                if not cost_ok:
                    pco_fallback_reason = 'invalid_costs'
                    pco_latency_ms = (time.perf_counter() - t_start) * 1000.0
                else:
                    pco_occupancy_valid = True
                    pco_valid_mode_fraction = diag['valid_mode_fraction']
                    finite_costs = costs[np.isfinite(costs)]
                    if len(finite_costs) > 0:
                        pco_cost_min = float(finite_costs.min())
                        pco_cost_mean = float(finite_costs.mean())
                        pco_cost_max = float(finite_costs.max())
                    pco_cost_selected_original = float(costs[planner_selected_mode_index])

                    # Candidate set: valid + not masked by SparseDriveV2's own rescorer
                    candidate = valid_modes & ~planner_rescore_mask
                    candidate_idx = np.where(candidate)[0]
                    pco_valid_candidate_count = int(len(candidate_idx))

                    best = None
                    pco_progress_diag = {}

                    if len(candidate_idx) == 0:
                        pco_fallback_reason = 'no_valid_candidate'
                    else:
                        if self.pco_rescore_topk > 0 and self.pco_rescore_topk < len(candidate_idx):
                            score_rank = np.argsort(-sigmoid_scores[candidate_idx])
                            keep = set(candidate_idx[score_rank[:self.pco_rescore_topk]])
                            candidate_idx = np.array(
                                [i for i in candidate_idx if i in keep], dtype=np.intp
                            )

                        pco_valid_candidate_count = int(len(candidate_idx))
                        if len(candidate_idx) > 0:
                            pco_clear_candidate_count = int(np.sum(
                                np.isfinite(costs[candidate_idx])
                                & (costs[candidate_idx] == 0.0)
                            ))

                        if len(candidate_idx) == 0:
                            pco_fallback_reason = 'no_valid_candidate'
                        elif self.pco_rescore_cost_mode == 'binary':
                            best, pco_fallback_reason, pco_progress_diag = select_pco_candidate(
                                scores=sigmoid_scores,
                                costs=costs,
                                candidate_idx=candidate_idx,
                                planner_trajs=planner_trajs,
                                original_mode_index=planner_selected_mode_index,
                                selection_policy=self.pco_selection_policy,
                                progress_tolerance_m=self.pco_progress_tolerance_m,
                            )
                            if best is not None:
                                pco_cost_selected_rescored = float(costs[best])
                            # ========== TRAJECTORY-SPACE HYSTERESIS ==========
                            # Only applies to progress-preserving veto; never
                            # contaminates binary-veto (V1 reference) arm.
                            if (best is not None
                                and self._hysteresis_traj_world is not None
                                and self.pco_selection_policy == 'progress_preserving_veto'):
                                prev_carla = self._world_xy_to_carla_local(
                                    self._hysteresis_traj_world, ego_transform
                                )
                                prev_planner = self._carla_local_to_planner(prev_carla)
                                # Progress-eligible candidates only (same δ as V2 tier-1)
                                orig_forward = planner_trajs[planner_selected_mode_index, -1, 1]
                                delta = self.pco_progress_tolerance_m
                                eligible_mask = (
                                    (costs[candidate_idx] == 0.0)
                                    & np.isfinite(costs[candidate_idx])
                                    & (planner_trajs[candidate_idx, -1, 1] >= orig_forward - delta)
                                )
                                eligible_idx = candidate_idx[eligible_mask]
                                if len(eligible_idx) > 0:
                                    safe_trajs = planner_trajs[eligible_idx]
                                    diff = safe_trajs - prev_planner.reshape(1, 6, 2)
                                    dist = np.mean(np.sqrt(np.sum(diff**2, axis=2)), axis=1)
                                    best_retained = eligible_idx[int(np.argmin(dist))]
                                    if best_retained != best:
                                        orig_traj = planner_trajs[planner_selected_mode_index]
                                        d_ret = float(np.mean(np.sqrt(np.sum((planner_trajs[best_retained] - orig_traj)**2, axis=1))))
                                        d_v2 = float(np.mean(np.sqrt(np.sum((planner_trajs[best] - orig_traj)**2, axis=1))))
                                        if d_ret <= d_v2 + self.pco_hysteresis_margin:
                                            best = best_retained
                                            pco_cost_selected_rescored = float(costs[best])
                                            pco_hysteresis_active = True
                                            self._hysteresis_retained_count += 1
                                            # Label honestly — retained is progress-eligible (same δ)
                                            pco_fallback_reason = 'filtered_viable'
                                            # Refresh progress + distance diagnostics for the retained mode
                                            pco_progress_selected = float(planner_trajs[best, -1, 1])
                                            pco_progress_eligible_count = int(np.sum(
                                                (costs[candidate_idx] == 0.0)
                                                & np.isfinite(costs[candidate_idx])
                                                & (planner_trajs[candidate_idx, -1, 1] >= orig_forward - delta)
                                            ))
                                            pco_traj_distance_selected = d_ret
                        else:
                            rescored = (
                                sigmoid_scores[candidate_idx]
                                - self.pco_rescore_lambda * costs[candidate_idx]
                            )
                            best_local = int(np.argmax(rescored))
                            best = candidate_idx[best_local]
                            pco_cost_selected_rescored = float(costs[best])
                            if costs[best] > 0.0 and np.all(costs[candidate_idx] > 0.0):
                                pco_fallback_reason = 'all_unsafe'
                                best = None
                            else:
                                pco_fallback_reason = 'filtered'

                    # Apply rescored trajectory to PID + logger
                    if best is not None and best != planner_selected_mode_index:
                        num_lon_mode = output['lon_reg'].shape[0]
                        row, col = divmod(best, num_lon_mode)
                        pco_selected_mode_rescored = best
                        pco_traj_rescored = (
                            output['traj_reg'][best].clone().cpu().numpy()
                        )
                        sparsedrive_traj = pco_traj_rescored.copy()
                        output['traj_final'] = output['traj_reg'][best].clone().cpu()
                        output['lat_reg_final'] = output['lat_reg'][row].clone().cpu()
                        output['lon_reg_final'] = output['lon_reg'][col].clone().cpu()
                        output['traj_selected_mode_index'] = best
                        planner_selected_mode_index = best
                        pco_mode_changed = True

                    # Progress diagnostics from selection helper
                    if pco_progress_diag:
                        pco_progress_original = pco_progress_diag.get('progress_original', pco_progress_original)
                        pco_progress_max_safe = pco_progress_diag.get('progress_max_safe', pco_progress_max_safe)
                        pco_progress_threshold = pco_progress_diag.get('progress_threshold', pco_progress_threshold)
                        pco_progress_filter_active = pco_progress_diag.get('progress_filter_active', pco_progress_filter_active)
                        # When hysteresis overrode V2 selection, the hysteresis block already
                        # refreshed these three fields for the retained mode.
                        if not pco_hysteresis_active:
                            pco_progress_selected = pco_progress_diag.get('progress_selected', pco_progress_selected)
                            pco_progress_eligible_count = pco_progress_diag.get('progress_eligible_count', pco_progress_eligible_count)
                            pco_traj_distance_selected = pco_progress_diag.get('traj_distance_selected', pco_traj_distance_selected)

                    # Safe-candidate rank: original-score rank of the safest candidate
                    if len(candidate_idx) > 0:
                        safest = candidate_idx[np.argmin(costs[candidate_idx])]
                        score_rank = np.argsort(-sigmoid_scores)
                        pco_safe_candidate_rank = int(
                            np.where(score_rank == safest)[0][0]
                        )

                    # Final cost: always reflects the actually executed mode
                    final_cost = costs[pco_selected_mode_rescored]
                    if np.isfinite(final_cost):
                        pco_cost_selected_rescored = float(final_cost)

                    pco_latency_ms = (time.perf_counter() - t_start) * 1000.0

                if cost_ok and self.pco_rescore_log_full_costs:
                    pco_full_costs = costs.astype(np.float32)

        if self.strict_campaign:
            arm = os.environ['HYDRIVE_ARM']
            if arm == 'baseline':
                assert not pco_mode_changed
                assert pco_selected_mode_rescored == pco_selected_mode_original
            elif pco_fallback_reason == 'filtered':
                assert pco_clear_candidate_count > 0
                assert np.isfinite(pco_cost_selected_rescored)
                assert pco_cost_selected_rescored == 0.0
            elif pco_fallback_reason == 'all_unsafe':
                assert pco_valid_candidate_count > 0
                assert pco_clear_candidate_count == 0
                assert pco_selected_mode_rescored == pco_selected_mode_original
            elif pco_fallback_reason == 'no_valid_candidate':
                assert pco_valid_candidate_count == 0
                assert pco_selected_mode_rescored == pco_selected_mode_original

        # ========== UPDATE HYSTERESIS STATE ==========
        # Store current trajectory in world coordinates for next frame's hysteresis.
        # Always store (whether or not intervention was enabled) so hysteresis
        # can follow the original planner trajectory when PCO is off.
        if not self.strict_campaign:
            self._hysteresis_traj_world = self._planner_to_carla_local(
                sparsedrive_traj
            )
            self._hysteresis_traj_world = self._local_to_world_xy(
                self._hysteresis_traj_world, ego_transform
            )
            if not pco_hysteresis_active:
                self._hysteresis_retained_count = 0

        # =========== LOG ROUTE DATA ==========
        if planner_trajs is not None and (
            self.strict_campaign or planner_scores is not None
        ):
            strict_frame_metadata = {
                'pco_fallback_reason': pco_fallback_reason,
                'pco_mode_changed': int(pco_mode_changed),
                'pco_selected_mode_original': int(pco_selected_mode_original),
                'pco_selected_mode_rescored': int(pco_selected_mode_rescored),
                'pco_cost_selected_original': pco_cost_selected_original,
                'pco_cost_selected_rescored': pco_cost_selected_rescored,
                'pco_valid_candidate_count': pco_valid_candidate_count,
                'pco_clear_candidate_count': pco_clear_candidate_count,
                'pco_occupancy_source': self.pco_occupancy_source,
                'pco_occupancy_valid': int(pco_occupancy_valid),
            }
            self.route_logger.log_timestep(
                planner_trajs=planner_trajs,
                planner_scores=planner_scores,
                planner_logits=planner_logits,
                planner_logits_masked=planner_logits_masked,
                planner_scores_masked=planner_scores_masked,
                planner_rescore_mask=planner_rescore_mask,
                planner_collision_rescore_mask=planner_collision_rescore_mask,
                planner_detection_rescore_mask=planner_detection_rescore_mask,
                planner_selected_mode_index=planner_selected_mode_index,
                traj_final=sparsedrive_traj,  # (6, 2) — decoder's selected trajectory after rescore
                occupancy_grid=occ_grid,
                ego_transform=ego_transform,
                ego_speed=ego_speed,
                collision=collision_occurred,              # bool
                collision_event=collision_event,
                simulator_frame=simulator_frame,
                simulator_elapsed_seconds=simulator_elapsed_seconds,
                agent_timestamp=timestamp,
                metadata=(strict_frame_metadata if self.strict_campaign else {
                    'route_step': self.step,
                    'command': command,
                    'guardian_intervene': guardian_intervene,
                    'occupancy_source': occ_meta.get('source', 'unknown'),
                    'obstacle_count': occ_meta.get('obstacle_count', 0),
                    'skipped_actor_count': occ_meta.get('skipped_actor_count', 0),
                    'planner_modes': planner_trajs.shape[0],
                    'ego_vehicle_length_m': self.guardian.vehicle_length,
                    'ego_vehicle_width_m': self.guardian.vehicle_width,
                    'ego_bbox_offset_forward_m': self.guardian._ego_bbox_location.x,
                    'ego_bbox_offset_right_m': self.guardian._ego_bbox_location.y,
                    'ego_bbox_rotation_yaw_deg': self.guardian._ego_bbox_rotation.yaw,
                    'guardian_longitudinal_margin_m': self.guardian.longitudinal_margin,
                    'guardian_lateral_margin_m': self.guardian.lateral_margin,
                    'guardian_sweep_spacing_m': self.guardian.sweep_sample_spacing,
                    # Schema-v5 PCO rescoring fields
                    'pco_rescore_enabled': int(self.pco_rescore_enabled),
                    'pco_lambda': self.pco_rescore_lambda,
                    'pco_rescore_cost_mode': self.pco_rescore_cost_mode,
                    'pco_rescore_score_source': self.pco_rescore_score_source,
                    'pco_rescore_topk': self.pco_rescore_topk,
                    'pco_fallback_reason': pco_fallback_reason,
                    'pco_mode_changed': int(pco_mode_changed),
                    'pco_selected_mode_original': int(pco_selected_mode_original),
                    'pco_selected_mode_rescored': int(pco_selected_mode_rescored),
                    'pco_traj_original': pco_traj_original.astype(np.float32),
                    'pco_traj_rescored': pco_traj_rescored.astype(np.float32),
                    'pco_cost_selected_original': pco_cost_selected_original,
                    'pco_cost_selected_rescored': pco_cost_selected_rescored,
                    'pco_cost_min': pco_cost_min,
                    'pco_cost_mean': pco_cost_mean,
                    'pco_cost_max': pco_cost_max,
                    'pco_valid_mode_fraction': pco_valid_mode_fraction,
                    'pco_valid_candidate_count': pco_valid_candidate_count,
                    'pco_clear_candidate_count': pco_clear_candidate_count,
                    'pco_safe_candidate_rank': pco_safe_candidate_rank,
                    'pco_latency_ms': pco_latency_ms,
                    # Schema-v6 temporal occupancy fields
                    'pco_occupancy_source': self.pco_occupancy_source,
                    'pco_temporal_waypoint_times': self.pco_temporal_waypoint_times,
                    'pco_temporal_dynamic_actor_count': pco_temporal_dynamic_actor_count,
                    'pco_temporal_static_actor_count': pco_temporal_static_actor_count,
                    'pco_temporal_skipped_actor_count': pco_temporal_skipped_actor_count,
                    'pco_temporal_build_latency_ms': pco_temporal_build_latency_ms,
                    'pco_temporal_valid': int(pco_temporal_valid),
                    'pco_occupancy_valid': int(pco_occupancy_valid),
                    # Progress-preserving veto fields (additive extension to schema v6)
                    'pco_selection_policy': pco_selection_policy_logged,
                    'pco_progress_tolerance_m': pco_progress_tolerance_m_logged,
                    'pco_progress_original': pco_progress_original,
                    'pco_progress_selected': pco_progress_selected,
                    'pco_progress_max_safe': pco_progress_max_safe,
                    'pco_progress_threshold': pco_progress_threshold,
                    'pco_progress_eligible_count': pco_progress_eligible_count,
                    'pco_progress_filter_active': int(pco_progress_filter_active),
                    'pco_traj_distance_selected': pco_traj_distance_selected,
                    'pco_hysteresis_active': int(pco_hysteresis_active),
                    'pco_hysteresis_retained_count': pco_hysteresis_retained_count,
                    'pco_planner_modes_lon': pco_planner_modes_lon,
                })
            )

            if self.pco_rescore_log_full_costs and pco_full_costs is not None:
                self.route_logger.timesteps[-1]['pco_full_costs'] = pco_full_costs
        else:
            raise RuntimeError(
                f"Missing planner trajectories or scores at step {self.step}; "
                "final-study logging cannot skip simulator frames"
            )
        
        # ========== NORMAL PID CONTROL ==========
        steer_traj, throttle_traj, brake_traj, metadata_traj = self.pidcontroller.control_pid(
            output, tick_data['speed'], local_command_xy
        )
        
        if brake_traj < 0.05: brake_traj = 0.0
        if throttle_traj > brake_traj: brake_traj = 0.0

        # ========== APPLY GUARDIAN OVERRIDE ==========
        if guardian_intervene:
            print(f"[Guardian] 🚨 INTERVENING at step {self.step} - Brake: {guardian_brake:.2f}")
            brake_traj = max(brake_traj, guardian_brake)
            throttle_traj = 0.0  # Cut throttle when braking
        
        control = carla.VehicleControl()
        self.pid_metadata = metadata_traj
        self.pid_metadata['agent'] = 'only_traj'
        
        control.steer = np.clip(float(steer_traj), -1, 1)
        control.throttle = np.clip(float(throttle_traj), 0, 1)
        control.brake = np.clip(float(brake_traj), 0, 1)     
        
        self.pid_metadata['steer'] = control.steer
        self.pid_metadata['throttle'] = control.throttle
        self.pid_metadata['brake'] = control.brake
        self.pid_metadata['steer_traj'] = float(steer_traj)
        self.pid_metadata['throttle_traj'] = float(throttle_traj)
        self.pid_metadata['brake_traj'] = float(brake_traj)
        self.pid_metadata['command'] = command
        self.pid_metadata['local_command_xy'] = local_command_xy
        
        # Add Guardian metadata
        self.pid_metadata['guardian_intervene'] = int(guardian_intervene)
        self.pid_metadata['guardian_brake'] = float(guardian_brake)
        
        self.result = output_data_batch[0]['img_bbox']
        self.result["control"] = control
        self.result["pid_metadata"] = self.pid_metadata
        
        metric_info = self.get_metric_info()
        self.metric_info[self.step] = metric_info
        
        if self.step % self.save_interval == 0:
            self.save(tick_data)

        # ========== STUCK-VEHICLE DETECTION ==========
        # After a collision, if the ego vehicle stays at near-zero velocity for
        # >100 timesteps (~5s), flush the route log and end the route early.
        # This prevents the scenario runner from logging 4000+ useless zero-velocity
        # frames while waiting for the route timeout.
        if self.early_truncation_enabled and self.collision_latched:
            if ego_speed < 0.1:
                if self._stuck_counter is None:
                    self._stuck_counter = 0
                self._stuck_counter += 1
                if self._stuck_counter > 100:
                    route_name = f"route_{self.save_name}"
                    print(f"[Stuck] Collision + zero velocity for >5s. Saving log ({self.step} steps) and ending route.")
                    self.route_logger.truncate_and_save(route_name, self.step)
                    try:
                        marker = f"close_loop_log/result/{self.save_name}.intentional_truncation"
                        os.makedirs(os.path.dirname(marker), exist_ok=True)
                        with open(marker, 'w') as f:
                            f.write(f"truncated_at_step_{self.step}")
                    except Exception as e:
                        print(f"[Stuck] Failed to write marker: {e}")
                    raise RuntimeError("Vehicle stuck after collision, ending route")
            else:
                self._stuck_counter = None
        
        return control
    
    def save(self, tick_data):
        if self.visualizer is None:
            return
        frame = self.step // self.save_interval
        self.data["index"] = self.step
        try:
            self.visualizer.add_vis(frame, self.data, self.result)
        except Exception:
            pass
        
        # metric info
        outfile = open(self.save_path / 'metric_info.json', 'w')
        json.dump(self.metric_info, outfile, indent=4)
        outfile.close()
    
    def destroy(self):
        """
        Cleans up the agent and all attached sensors safely.
        """
        # 1. DATA PRESERVATION (First priority)
        try:
            route_name = f"route_{self.save_name}"
            self.route_logger.save_route(route_name)
        except Exception as e:
            print(f"[Agent] Failed to save route logs: {e}")
        
        # 2. SENSOR CLEANUP
        if hasattr(self, 'collision_sensor') and self.collision_sensor is not None:
            try:
                self.collision_sensor.destroy()
            except RuntimeError as e:
                # Expected if CARLA world is already shutting down
                print(f"[Agent] Sensor cleanup (expected during shutdown): {e}")
            except Exception as e:
                print(f"[Agent] Error during sensor cleanup: {e}")
            finally:
                self.collision_sensor = None

        # 3. STATE RESET
        self.collision_latched = False

        # 4. MEMORY MANAGEMENT (Preventing GPU/RAM leaks)
        if hasattr(self, 'model'):
            try:
                # Move to CPU before deleting to ensure VRAM is released properly
                self.model.cpu()
                del self.model
            except Exception:
                pass
        
        # Force CUDA to drop the cleared tensors
        torch.cuda.empty_cache()

        # 5. VISUALIZATION EXPORT
        # Do this last as it's the most CPU-intensive non-CARLA task
        # (getattr: destroy() is also called when setup() failed before the
        # visualizer attribute was ever assigned)
        if getattr(self, 'visualizer', None) is not None:
            try:
                self.visualizer.image2video()
            except Exception as e:
                print(f"[Agent] Video export failed: {e}")


    
    def gps_to_location(self, gps):
        EARTH_RADIUS_EQUA = 6378137.0
        lat, lon = gps
        scale = math.cos(self.lat_ref * math.pi / 180.0)
        my = math.log(math.tan((lat+90) * math.pi / 360.0)) * (EARTH_RADIUS_EQUA * scale)
        mx = (lon * (math.pi * EARTH_RADIUS_EQUA * scale)) / 180.0
        y = scale * EARTH_RADIUS_EQUA * math.log(math.tan((90.0 + self.lat_ref) * math.pi / 360.0)) - my
        x = mx - scale * self.lon_ref * math.pi * EARTH_RADIUS_EQUA / 180.0
        return np.array([x, y])
    
    def get_augmentation(self):
        H = 900 * resize_scale
        W = 1600 * resize_scale
        fH, fW = self.data_aug_conf["final_dim"]
        resize = max(fH / H, fW / W)
        resize_dims = (int(W * resize), int(H * resize))
        newW, newH = resize_dims
        crop_h = (
            int((1 - np.mean(self.data_aug_conf["bot_pct_lim"])) * newH)
            - fH
        )
        crop_w = int(max(0, newW - fW) / 2)
        crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
        flip = False
        rotate = 0
        rotate_3d = 0
        
        aug_config = {
            "resize": resize,
            "resize_dims": resize_dims,
            "crop": crop,
            "flip": flip,
            "rotate": rotate,
            "rotate_3d": rotate_3d,
        }
        return aug_config
    
    def invert_pose(self, pose):
        inv_pose = np.eye(4)
        inv_pose[:3, :3] = np.transpose(pose[:3, :3])
        inv_pose[:3, -1] = - inv_pose[:3, :3] @ pose[:3, -1]
        return inv_pose
    
    def get_metric_info(self):
        # Implement this based on your needs
        return {}
