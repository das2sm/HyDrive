import os
import sys
import zmq
import pickle
import torch
import numpy as np
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet3d.models import build_model
from mmcv.utils import import_modules_from_strings

# This force-registers SparseDrive into the MMDetection3D registry
try:
    import_modules_from_strings(['projects.mmdet3d_plugin'])
    print("[+] Successfully registered SparseDrive plugins.")
except ImportError as e:
    print(f"[!] Failed to register plugins: {e}")

# --- CONFIGURATION & PATHS ---
SPARSEDRIVE_PATH = "/workspace/SparseDrive"
sys.path.insert(0, SPARSEDRIVE_PATH)
sys.path.insert(0, os.path.join(SPARSEDRIVE_PATH, "projects"))

CONFIG_PATH = os.path.join(SPARSEDRIVE_PATH, "projects/configs/sparsedrive_small_stage2.py")
CHECKPOINT_PATH = os.path.join(SPARSEDRIVE_PATH, "ckpt/sparsedrive_stage2.pth")

# --- MODEL INITIALIZATION ---
def init_model():
    print(f"[*] Initializing SparseDrive from: {CONFIG_PATH}")
    
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"Config not found: {CONFIG_PATH}")
    print(f"[✓] Config file exists")
    
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")
    print(f"[✓] Checkpoint file exists")
    
    cfg = Config.fromfile(CONFIG_PATH)
    
    kmeans_base_path = os.path.join(SPARSEDRIVE_PATH, "data/kmeans")
    
    if not os.path.exists(kmeans_base_path):
        raise FileNotFoundError(f"Kmeans directory not found at: {kmeans_base_path}")
    print(f"[✓] Kmeans directory exists")
    
    cfg_dict = cfg.to_dict()
    
    def recursive_patch(d, path=""):
        if isinstance(d, dict):
            for k, v in d.items():
                current_path = f"{path}.{k}" if path else k
                if k in ['anchor', 'motion_anchor', 'plan_anchor'] and isinstance(v, str) and 'kmeans' in v:
                    filename = os.path.basename(v)
                    new_path = os.path.join(kmeans_base_path, filename)
                    
                    if not os.path.exists(new_path):
                        raise FileNotFoundError(f"Required kmeans file not found: {new_path}")
                    
                    print(f"[*] Patching: {current_path}")
                    print(f"    {v} -> {new_path}")
                    d[k] = new_path
                else:
                    recursive_patch(v, current_path)
        elif isinstance(d, list):
            for idx, item in enumerate(d):
                recursive_patch(item, f"{path}[{idx}]")
    
    print("[*] Scanning config for kmeans paths...")
    recursive_patch(cfg_dict)
    
    cfg = Config(cfg_dict)
    
    print(f"[*] Building model...")
    model = build_model(cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
    
    print(f"[*] Loading checkpoint weights...")
    load_checkpoint(model, CHECKPOINT_PATH, map_location='cpu')
    
    print(f"[*] Moving model to GPU...")
    model.cuda()
    model.eval()
    print("[✓] Model ready!")
    return model

# --- PREPROCESSING ---
def preprocess_images(image_dict):
    view_names = [
        'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
        'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
    ]
    
    mean = np.array([123.675, 116.28, 103.53])
    std = np.array([58.395, 57.12, 57.375])
    
    processed_imgs = []
    for name in view_names:
        img = image_dict[name]
        img = (img - mean) / std
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        processed_imgs.append(img)
        
    img_tensor = torch.stack(processed_imgs).unsqueeze(0).cuda()
    return img_tensor

def create_img_metas(timestamp=0.0, ego_matrix=None):
    fx = 704.0
    fy = 704.0
    cx = 352.0
    cy = 128.0
    
    intrinsic = np.array([
        [fx,  0,  cx, 0],
        [0,  fy,  cy, 0],
        [0,   0,   1, 0],
        [0,   0,   0, 1]
    ], dtype=np.float32)
    
    extrinsic = np.eye(4, dtype=np.float32)
    projection_mat = intrinsic @ extrinsic
    projection_mats = np.stack([projection_mat] * 6, axis=0)
    
    if ego_matrix is not None:
        t_global = np.array(ego_matrix, dtype=np.float32)
    else:
        t_global = np.eye(4, dtype=np.float32)

    # Calculate the Inverse (World -> Ego)
    # This is critical for the "Memory" of the model to work
    try:
        t_global_inv = np.linalg.inv(t_global)
    except:
        t_global_inv = np.eye(4, dtype=np.float32)

    command = np.array([0, 0, 1], dtype=np.float32)

    return [{
        'box_type_3d': None,
        'can_bus': np.zeros(18, dtype=np.float32),
        'lidar2img': [np.eye(4, dtype=np.float32) for _ in range(6)],
        'projection_mat': projection_mats,
        'img_shape': [(256, 704)] * 6,
        'ori_shape': [(256, 704)] * 6,
        'pad_shape': [(256, 704)] * 6,
        'timestamp': timestamp,
        'T_global': t_global,       
        'T_global_inv': t_global_inv,
        'gt_ego_fut_cmd': command
    }]

# --- MAIN SERVER ---
def main():
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:5555")
    print("[+] ZMQ Server bound to port 5555")

    try:
        model = init_model()
        print("[+] Model loaded onto GPU. Ready for CARLA frames.")
    except Exception as e:
        print(f"[!] Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return

    frame_count = 0
    while True:
        try:
            message = socket.recv()
            data_in = pickle.loads(message)

            # 1. Image Preprocessing
            image_dict = data_in['images']
            img_tensor = preprocess_images(image_dict)
            frame_count += 1
            
            # 2. Prepare Metadata (Correctly passing the matrix from CARLA)
            # This creates the list structure SparseDrive expects internally
            metas_list = create_img_metas(
                timestamp=data_in.get('timestamp', 0.0),
                ego_matrix=data_in.get('ego_matrix') # This is the 4x4 list from sim.py
            )
            
            # 3. Flatten metadata for the model's **kwargs
            # We take the first dict but keep the list reference inside it
            img_metas_dict = metas_list[0]
            img_metas_dict['img_metas'] = metas_list 

            # 4. Convert all necessary fields to GPU Tensors
            # This replaces your unused timestamp_tensor logic
            img_metas_dict['timestamp'] = torch.tensor([data_in.get('timestamp', 0.0)]).cuda().float()
            
            img_metas_dict['projection_mat'] = torch.from_numpy(
                img_metas_dict['projection_mat']).cuda().float().unsqueeze(0)

            # Command intent (Go Straight)
            img_metas_dict['gt_ego_fut_cmd'] = torch.from_numpy(
                img_metas_dict['gt_ego_fut_cmd']).cuda().float().unsqueeze(0)

            if 'lidar2img' in img_metas_dict:
                img_metas_dict['lidar2img'] = torch.from_numpy(
                    np.stack(img_metas_dict['lidar2img'])).cuda().float().unsqueeze(0)

            # 5. Run Inference
            with torch.no_grad():
                outputs = model(
                    img=img_tensor, 
                    return_loss=False, 
                    **img_metas_dict # Unpacks timestamp, projection_mat, etc.
                )

            print(f"[DEBUG] Frame {frame_count} - Model output keys: {outputs[0].keys()}")
            
            res = outputs[0]
            planning_data = None

            if 'sd_planning' in res:
                planning_data = res['sd_planning']
            elif 'planning' in res:
                planning_data = res['planning']
            elif 'img_bbox' in res and 'final_planning' in res['img_bbox']:
                # This is the most likely location in SparseDrive stage2
                planning_data = res['img_bbox']['final_planning']
            elif 'img_bbox' in res and 'planning' in res['img_bbox']:
                planning_data = res['img_bbox']['planning']

            if planning_data is not None:
                # planning_data is usually a tensor of shape (6, 2) or (1, 6, 2)
                if isinstance(planning_data, torch.Tensor):
                    raw_waypoints = planning_data.detach().cpu().numpy()
                else:
                    # If it's the list returned by the decoder: [{'planning': ...}]
                    raw_waypoints = planning_data[0]['final_planning'].numpy()

                # Ensure it's (6, 2)
                if raw_waypoints.ndim == 3: # (1, 6, 2)
                    raw_waypoints = raw_waypoints[0]
                
                waypoints = raw_waypoints.tolist()
                print(f"[*] Successfully extracted {len(waypoints)} real waypoints.")
            else:
                print("[!] Warning: Planning results not found in any known keys.")
                print(f"[DEBUG] Keys in img_bbox: {list(res['img_bbox'].keys()) if 'img_bbox' in res else 'N/A'}")
                waypoints = [[0, 0]] * 6

            print(f"[*] Frame {data_in.get('timestamp', '??')}: Inference successful. Sending {len(waypoints)} waypoints.")

            response = {
                'waypoints': waypoints,
                'status': 'ok',
                'confidence': 1.0
            }
            socket.send(pickle.dumps(response))

        except KeyboardInterrupt:
            print("\n[!] Shutting down inference server...")
            break
        except Exception as e:
            print(f"[!] Runtime error: {e}")
            import traceback
            traceback.print_exc()
            error_response = {
                'waypoints': [],
                'status': 'error',
                'confidence': 0.0
            }
            socket.send(pickle.dumps(error_response))

if __name__ == "__main__":
    main()