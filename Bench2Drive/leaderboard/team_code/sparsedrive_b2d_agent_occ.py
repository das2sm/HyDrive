import os
import json
import datetime
import pathlib
import time
import copy
import math
from scipy.optimize import fsolve
from pyquaternion import Quaternion
from PIL import Image
import cv2
import numpy as np
import torch
import pickle
import carla

from team_code.pid_controller import PIDController
from team_code.planner import RoutePlanner
from team_code.guardian import Guardian  # ← GUARDIAN IMPORT
from team_code.divergence_logger import DivergenceLogger  # ← DIVERGENCE LOGGER IMPORT

from leaderboard.autoagents import autonomous_agent
from leaderboard.utils.route_manipulation import _get_latlon_ref
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from mmcv import Config
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.parallel.collate import collate as mm_collate_to_batch_form
from mmdet.models import build_detector
from mmdet.datasets.pipelines import Compose

from tools.visualization.visualize import Visualizer

IS_BENCH2DRIVE = os.environ.get('IS_BENCH2DRIVE', None)
CAMERAS = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']

frame_rate = 10
resize_scale = 0.44
save_interval = 200

lefthand_ego_to_lidar = np.array([[ 0, 1, 0, 0],
                                  [ 1, 0, 0, 0],
                                  [ 0, 0, 1, 0],
                                  [ 0, 0, 0, 1]])
left2right = np.eye(4)
left2right[1,1] = -1


def get_entry_point():
    return 'SparseDriveAgent'


class Clock():
    def __init__(self):
        self.time =  time.time()
        self.verbose = False
    
    def count(self, tag):
        if self.verbose:
            prev_time = self.time
            self.time = time.time()
            print(tag, self.time - prev_time)
        else:
            pass

    
class SparseDriveAgent(autonomous_agent.AutonomousAgent):
    def setup(self, path_to_conf_file):
        self.track = autonomous_agent.Track.SENSORS
        self.steer_step = 0
        self.last_moving_status = 0
        self.last_moving_step = -1
        self.last_steer = 0
        
        config_parts = path_to_conf_file.split('+')
        self.config_path = config_parts[0]
        self.ckpt_path = config_parts[1]
        self.save_name = config_parts[2] if len(config_parts) >= 3 else "default_eval"
        self.gpu_rank = int(config_parts[3]) if len(config_parts) >= 4 else 0
        
        self.step = -1
        self.wall_start = time.time()
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
                    _module_dir = os.path.dirname(args.config)
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
        
        self.save_path = pathlib.Path(f'close_loop_log/save/{self.save_name}')
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
        
        self.visualizer = Visualizer(plot_choices, self.save_path, planning_key=cfg.get("anchor_reference_group", "spatial"))
        
        # ========== GUARDIAN INITIALIZATION ==========
        self.use_guardian = False  # Toggle to enable/disable Guardian
        if self.use_guardian:
            self.guardian = Guardian(
                world=None,  # Will be set in _init()
                log_dir=str(self.save_path / 'guardian_logs'),
                debug=True  # Set True to visualize in CARLA
            )
            print("[SparseDrive] Guardian initialized (oracle mode)")
        else:
            self.guardian = None
        # =============================================

        # ========== DIVERGENCE LOGGER ==========
        self.divergence_logger = DivergenceLogger(
            log_dir=str(self.save_path / 'divergence_logs'),
            horizon_seconds=3.0,
            fps=20
        )
        print("[SparseDrive] Divergence logger initialized")
        # =======================================
   
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
        
        self.clock = Clock() 
        self.stuck_detector = 0
        self.forced_move = 0
    
    def _init(self):
        self.lat_ref, self.lon_ref = _get_latlon_ref(CarlaDataProvider.get_world()) 
        
        self._route_planner = RoutePlanner(4.0, 50.0, lat_ref=self.lat_ref, lon_ref=self.lon_ref)
        self._route_planner.set_route(self._global_plan, True)
        
        self._route_planner_far = RoutePlanner(4.0, 50.0, lat_ref=self.lat_ref, lon_ref=self.lon_ref)
        self._route_planner_far.set_route(self._global_plan_far, True)
        
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
        
        collision_occurred = False
        collision_type = None      # store type identifier: 'vehicle', 'pedestrian', 'layout'
        
        if 'COLLISION' in input_data:
            # input_data['COLLISION'] is a list of CollisionEvent for this frame
            for collision_event in input_data['COLLISION']:
                collision_occurred = True
                # Get type info from collision_event.other_actor
                other = collision_event.other_actor
                if other is not None:
                    type_id = other.type_id
                    if 'vehicle' in type_id:
                        collision_type = 'vehicle'
                        break
                    elif 'walker' in type_id:
                        collision_type = 'pedestrian'
                        break
                    else:
                        collision_type = 'layout'
                else:
                    collision_type = 'layout'
        
        # Append to result dict
        result['collision'] = collision_occurred
        result['collision_type'] = collision_type
        return result
    
    @torch.no_grad()
    def run_step(self, input_data, timestamp):
        if not self.initialized:
            self._init()
        
        self.clock.count("start")
        tick_data = self.tick(input_data)
        self.clock.count("tick")
        
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
        
        self.clock.count("data")
        
        output_data_batch = self.model(**input_data_batch)
        self.clock.count("model")
        
        output = output_data_batch[0]['img_bbox']
         
        '''
        # DEBUGGING OUTPUT STRUCTURE - UNCOMMENT TO INSPECT MODEL OUTPUT
        for key, value in output.items():
            if hasattr(value, 'shape'):
                print(f"{key}: Type={type(value)}, Shape={value.shape}")
            else:
                print(f"{key}: Type={type(value)}, Value={value}")
        return
        '''

        # ========== EXTRACT SPARSEDRIVE TRAJECTORY ==========
        # SparseDrive outputs trajectory in 'traj_final'
        if 'traj_final' in output:
            sparsedrive_traj = output['traj_final']
            
            if torch.is_tensor(sparsedrive_traj):
                sparsedrive_traj = sparsedrive_traj.cpu().numpy()
            
            # Debug trajectory
            if self.step % 50 == 0 or self.step < 5:
                print(f"\n[DEBUG Step {self.step}]")
                print(f"Guardian trajectory shape: {sparsedrive_traj.shape}")
                print(f"First 3 waypoints:\n{sparsedrive_traj[:3]}")
                print(f"Max forward extent: {sparsedrive_traj[:, 1].max():.2f}m")
                print(f"Ego speed: {tick_data['speed']:.2f} m/s")

        else:
            print(f"[ERROR] 'traj_final' not found in output keys: {output.keys()}")

        # ========== EXTRACT EGO MULTI-MODAL PLANNER DISTRIBUTION ==========
        if 'traj_reg' not in output or 'traj_cls' not in output:
            print(f"[ERROR] Planner trajectory or scores not found in output. Keys: {output.keys()}")

        else:   
            K = 6
            planner_trajs = output['traj_reg']          # (1024, 6, 2)
            planner_scores = output['traj_cls']         # (1024,)
            topk_scores, topk_indices = torch.topk(planner_scores, k=K)
            planner_trajs = planner_trajs[topk_indices] # (6, 6, 2)
            planner_scores = topk_scores
            planner_scores = planner_scores / (planner_scores.sum() + 1e-8)   # normalize
            planner_trajs = planner_trajs.detach().cpu().numpy()  # (6, 6, 2)
            planner_scores = planner_scores.detach().cpu().numpy()  # (6,)

            assert planner_trajs.shape == (K, 6, 2), f"Wrong shape: {planner_trajs.shape}"

            if self.step % 50 == 0:
                print("\n[Divergence Debug]")
                print(f"Planner modes: {planner_trajs.shape[0]}")
                print(f"Waypoints per mode: {planner_trajs.shape[1]}")  
                print(f"Scores: {planner_scores}")
                # print("Raw traj_reg[0, :, :]:\n", output['traj_reg'][0, :, :])
                
                # Show full trajectory for Mode 0
                print(f"\n✓ Mode 0 (score={planner_scores[0]:.3f}) - FULL TRAJECTORY:")
                for t in range(planner_trajs.shape[1]):
                    wp = planner_trajs[0, t]
                    print(f"  t={t}: [{wp[0]:6.2f}, {wp[1]:6.2f}] (left, forward)")
                
                # Show endpoints for all modes
                print(f"\n✓ All mode endpoints:")
                for i in range(K):
                    endpoint = planner_trajs[i, -1]
                    print(f"  Mode {i} (score={planner_scores[i]:.3f}): [{endpoint[0]:6.2f}, {endpoint[1]:6.2f}]")

        # ========== GET EGO STATE FOR GUARDIAN ==========
        ego_actor = CarlaDataProvider.get_hero_actor()
        ego_transform = ego_actor.get_transform()
        ego_speed = tick_data['speed']
        
        # ========== COMPUTE OCCUPANCY AND BASELINE SIGNALS ==========
        if self.guardian is not None:
            # Get occupancy grid from Guardian
            occ_grid, occ_meta = self.guardian.build_carla_occupancy(ego_transform)
            
            # Compute minimum distance to forward obstacles
            occupied = np.argwhere(occ_grid > 0.5)
            if len(occupied) > 0:
                local_occ = self.guardian._grid_to_local(occupied)
                # Only consider forward obstacles (forward = positive in first column)
                forward_obstacles = local_occ[local_occ[:, 0] > 0.0]
                if len(forward_obstacles) > 0:
                    min_dist = float(np.linalg.norm(forward_obstacles, axis=1).min())
                else:
                    min_dist = 999.0
            else:
                min_dist = 999.0
            
            # Compute TTC (simplified)
            if ego_speed > 0.5 and min_dist < 50:
                ttc = min_dist / max(ego_speed, 0.1)
            else:
                ttc = 999.0
        else:
            # No Guardian - create dummy occupancy
            occ_grid = np.zeros((240, 240), dtype=np.float32)
            min_dist = 999.0
            ttc = 999.0
            occ_meta = {'source': 'none', 'actor_count': 0}

        # ========== APPLY GUARDIAN INTERVENTION ==========
        guardian_intervene = False
        guardian_brake = 0.0
        
        if self.guardian is not None and self.use_guardian:
            try:
                guardian_intervene, guardian_brake = self.guardian.evaluate(
                    traj=sparsedrive_traj,
                    ego_transform=ego_transform,
                    speed=ego_speed,
                    ego_actor=ego_actor
                )
            except Exception as e:
                print(f"[Guardian ERROR] {e}")
                import traceback
                traceback.print_exc()
        
        # ========== DETECT COLLISION/NEAR-MISS ==========
        '''
        collision_occurred = tick_data.get('collision', False)
        collision_type = tick_data.get('collision_type', None)
        '''

        # TODO
        collision_occurred = False
        near_miss = False

        '''
        near_miss = (min_dist < 1.5) and (ego_speed > 1.0) and (not collision_occurred)

        if collision_occurred:
            print(f"Collision detected with {collision_type}")
        elif near_miss:
            print("Near miss detected: Obstacle within 1.5m at speed > 1.0 m/s")    
        '''
        # ========== LOG DIVERGENCE DATA ==========
        self.divergence_logger.log_timestep(
            planner_trajs=planner_trajs,      # (K, T, 2) - multiple trajectory modes
            planner_scores=planner_scores,       # (K,) - probability weights
            occupancy_grid=occ_grid,          # (H, W) - BEV occupancy
            ego_transform=ego_transform,      # CARLA transform
            ego_speed=ego_speed,              # float
            ttc=ttc,                          # float
            min_distance=min_dist,            # float
            collision=collision_occurred,              # bool
            near_miss=near_miss,              # bool
            metadata={
                'route_step': self.step,
                'command': command,
                'guardian_intervene': guardian_intervene,
                'occupancy_source': occ_meta.get('source', 'unknown'),
                'num_actors': occ_meta.get('actor_count', 0),
                'planner_modes': planner_trajs.shape[0]
            }
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
        
        # ========== SAVE GUARDIAN LOG ==========
        if self.guardian is not None:
            self.guardian.save_log(
                step=self.step,
                timestamp=timestamp,
                throttle_cmd=control.throttle
            )
        
        return control
    
    def save(self, tick_data):
        frame = self.step // self.save_interval
        self.data["index"] = self.step
        self.visualizer.add_vis(frame, self.data, self.result)
        
        # metric info
        outfile = open(self.save_path / 'metric_info.json', 'w')
        json.dump(self.metric_info, outfile, indent=4)
        outfile.close()
    
    def destroy(self):
        # Save divergence log before cleanup
        route_name = f"route_{self.save_name}"
        self.divergence_logger.save_route(route_name)

        del self.model
        torch.cuda.empty_cache()
        self.visualizer.image2video()
    
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