from time import time

import carla
import numpy as np
import zmq
import pickle
import queue

# --- 1. SENSOR SYNC CLASS ---
class SensorGrabber:
    def __init__(self, names):
        self.queues = {name: queue.Queue() for name in names}
        self.names = names

    def callback(self, image, name):
        self.queues[name].put(image)

    def get_all_images(self, timeout=2.0):
        data = {}
        for name in self.names:
            try:
                img = self.queues[name].get(timeout=timeout)
                array = np.frombuffer(img.raw_data, dtype=np.uint8)
                array = array.reshape((img.height, img.width, 4))
                data[name] = array[:, :, :3]
            except queue.Empty:
                return None
        return data


def nuscenes_to_carla(waypoints):
    """
    Correct Mapping:
    nuScenes Y (Forward) -> CARLA X (Forward)
    nuScenes X (Right)   -> CARLA Y (Right)
    """
    # wp[0] is nuScenes X (Right), wp[1] is nuScenes Y (Forward)
    return [[wp[1], wp[0]] for wp in waypoints]


# --- 2. SETUP ---
context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:5555")

client = carla.Client("localhost", 2000)
client.set_timeout(20.0)
world = client.get_world()

# Cleanup & Sync
for a in world.get_actors().filter('vehicle.*'): a.destroy()
for a in world.get_actors().filter('sensor.*'): a.destroy()

settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.1
world.apply_settings(settings)

# Spawn Tesla
bp = world.get_blueprint_library().filter("vehicle.tesla.model3")[0]
spawn_point = world.get_map().get_spawn_points()[0]
vehicle = world.spawn_actor(bp, spawn_point)

# --- 3. ATTACH 6 CAMERAS ---
camera_bp = world.get_blueprint_library().find("sensor.camera.rgb")
camera_bp.set_attribute("image_size_x", "704")
camera_bp.set_attribute("image_size_y", "256")

camera_transforms = {
    "CAM_FRONT": carla.Transform(carla.Location(x=1.5, z=1.5), carla.Rotation(yaw=0)),
    "CAM_FRONT_LEFT": carla.Transform(carla.Location(x=1.3, y=-0.5, z=1.5), carla.Rotation(yaw=-55)),
    "CAM_FRONT_RIGHT": carla.Transform(carla.Location(x=1.3, y=0.5, z=1.5), carla.Rotation(yaw=55)),
    "CAM_BACK": carla.Transform(carla.Location(x=-1.5, z=1.5), carla.Rotation(yaw=180)),
    "CAM_BACK_LEFT": carla.Transform(carla.Location(x=-1.3, y=-0.5, z=1.5), carla.Rotation(yaw=-110)),
    "CAM_BACK_RIGHT": carla.Transform(carla.Location(x=-1.3, y=0.5, z=1.5), carla.Rotation(yaw=110)),
}

grabber = SensorGrabber(camera_transforms.keys())
camera_list = []
for name, transform in camera_transforms.items():
    cam = world.spawn_actor(camera_bp, transform, attach_to=vehicle)
    cam.listen(lambda image, n=name: grabber.callback(image, n))
    camera_list.append(cam)

# --- 4. MAIN LOOP ---
try:
    while True:
        world.tick()
        
        image_data = grabber.get_all_images()
        
        transform = vehicle.get_transform()
        ego_matrix = transform.get_matrix()
        
        if image_data is None:
            print("Waiting for camera sensors to fire...")
            continue

        start_time = time()
        
        socket.send(pickle.dumps({
            "images": image_data,
            "timestamp": world.get_snapshot().timestamp.elapsed_seconds,
            "can_bus": np.zeros(18),
            "ego_matrix": ego_matrix
        }))
        
        model_reply = socket.recv()
        data = pickle.loads(model_reply)
        
        # DEBUG: Print what we received
        print(f"[DEBUG] Received data keys: {data.keys()}")
        print(f"[DEBUG] Status: {data.get('status')}, Confidence: {data.get('confidence')}")
        
        waypoints = data.get('waypoints', [])
        print(f"[DEBUG] Received {len(waypoints)} waypoints")
        if waypoints:
            print(f"[DEBUG] First 2 waypoints: {waypoints[:2]}")
        
        # Check if we have valid waypoints
        if data.get('status') != 'ok' or len(waypoints) != 6:
            print("WARNING: Model error or invalid waypoints! BRAKING.")
            vehicle.apply_control(carla.VehicleControl(hand_brake=True, throttle=0.0))
        else:
            # Convert from nuScenes to CARLA coordinates
            carla_waypoints = nuscenes_to_carla(waypoints)
            print(f"[INFO] Valid waypoints received. First waypoint: {carla_waypoints[0]}")
            
            # TODO: Implement PID controller to follow these waypoints
            # For now, just let it idle
            vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0))

        print("Reply received from Model Server.")

        end_time = time()
        latency_ms = (end_time - start_time) * 1000
        print(f"Round-trip Latency: {latency_ms:.2f}ms | FPS: {1000/latency_ms:.1f}")
        
        spectator = world.get_spectator()
        v_trans = vehicle.get_transform()
        spectator.set_transform(carla.Transform(v_trans.location + carla.Location(z=25), carla.Rotation(pitch=-90)))

        print(f"Frame {world.get_snapshot().frame} synced with Model Server.")

finally:
    settings = world.get_settings()
    settings.synchronous_mode = False
    world.apply_settings(settings)
    for cam in camera_list:
        cam.stop()
        cam.destroy()
    vehicle.destroy()