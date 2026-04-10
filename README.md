This system creates a bridge between the **CARLA Simulator** (high-fidelity driving environment) and the **SparseDrive Model** (a state-of-the-art end-to-end autonomous driving transformer). 

The architecture uses a **Producer-Consumer** pattern connected via **ZeroMQ (ZMQ)** for low-latency communication.

---

### 1. `sim.py` (The CARLA Client & Sensor Interface)
This script acts as the "eyes and ears" of the car. It manages the simulation, collects data, and applies movement commands.

#### **How it works:**
* **Sensor Syncing:** It spawns 6 cameras (Front, Front-Left, Front-Right, Back, Back-Left, Back-Right) to match the nuScenes layout used by SparseDrive. It uses a `SensorGrabber` class with queues to ensure that frames from all 6 cameras are synchronized before being sent.
* **Ego-Motion Tracking:** Every frame, it extracts the `ego_matrix` (4x4 transform) from CARLA. This is vital for the model to understand its own velocity and orientation.
* **Data Packaging:** It pickles the 6 images, the timestamp, and the ego-matrix into a single bundle and sends it to the server via a `ZMQ.REQ` (Request) socket.
* **Coordinate Transformation:** It handles the translation between CARLA and the model. 
    * **The Nuance:** The model predicts in nuScenes coordinates. `sim.py` converts these back into CARLA coordinates so the steering and throttle can be applied correctly.

#### **What the Outputs Mean:**
* **Latency/FPS:** Measures how fast the entire loop (Sensor -> Model -> Control) is running. ~9 FPS is the current baseline.
* **First Waypoint:** Represents the target coordinate the car wants to reach in ~0.5 seconds. For example, `[2.5, -0.1]` means "move 2.5 meters forward and 0.1 meters to the right."

---

### 2. `inference_server.py` (The AI Brain)
This script hosts the SparseDrive model in a persistent state on the GPU, waiting to process data bundles from the simulator.

#### **How it works:**
* **Model Hosting:** It initializes the SparseDrive Stage-2 model, loading the weights into VRAM. It keeps an **InstanceBank** in memory, which allows the model to "remember" objects and movement across multiple frames.
* **Metadata Generation:** End-to-end models require complex "metadata" (camera intrinsics, projection matrices, and temporal transforms). The server generates these on-the-fly based on the `ego_matrix` provided by CARLA.
* **Temporal Math:** It calculates $T_{global\_inv}$ (the inverse global transform). This tells the model: *"Since the last frame, the car moved X meters forward; therefore, move all previous object detections back X meters in our memory so they stay aligned."*
* **Inference:** It runs the forward pass through the Transformer. It extracts the **Planning Head** output, which contains the predicted future path of the vehicle.

#### **What the Outputs Mean:**
* **img_bbox:** The model detects 3D boxes for all surrounding cars/pedestrians.
* **final_planning:** A set of 6 waypoints (typically representing the next 3 seconds of travel). These are the $(x, y)$ coordinates the model believes are the safest and most efficient path forward.
* **Status/Confidence:** `status: ok` confirms the model successfully produced a trajectory rather than crashing or timing out.

---

### 3. Data Flow Summary

| Step | Action | Responsibility | Data Transferred |
| :--- | :--- | :--- | :--- |
| **1** | Capture | `sim.py` | 6x RGB Images + Ego Transform |
| **2** | Send | **ZMQ Socket** | Pickled Data Packet |
| **3** | Process | `inference_server.py` | Preprocessing -> Transformer Forward Pass |
| **4** | Decode | `inference_server.py` | Extracting Waypoints from `img_bbox` |
| **5** | Reply | **ZMQ Socket** | Predicted Waypoints (6, 2) |
| **6** | Act | `sim.py` | Coordinate Flip -> Controller -> `apply_control()` |

### Key Coordination Points
* **nuScenes vs CARLA:** The model thinks $Y$ is forward; CARLA thinks $X$ is forward. The conversion happens in `sim.py` using `nuscenes_to_carla`.
* **Stage-2 Significance:** Because this is "Stage-2," the model isn't just detecting objects; it is actively planning a path based on those detections, making it a true "end-to-end" driving system.
