from collections import deque
from typing import List
import numpy as np
import torch

class PID(object):
    def __init__(self, K_P=1.0, K_I=0.0, K_D=0.0, n=20):
        self._K_P = K_P
        self._K_I = K_I
        self._K_D = K_D

        self._window = deque([0 for _ in range(n)], maxlen=n)
        self._max = 0.0
        self._min = 0.0

    def step(self, error):
        self._window.append(error)
        self._max = max(self._max, abs(error))
        self._min = -abs(self._max)
        if len(self._window) >= 2:
            integral = np.mean(self._window)
            derivative = (self._window[-1] - self._window[-2])
        else:
            integral = 0.0
            derivative = 0.0

        return self._K_P * error + self._K_I * integral + self._K_D * derivative



class PIDController(object):
    
    def __init__(self, pid_config, sample_interval=10, max_throttle=1.0, brake_speed=0.2,brake_ratio=1.1, clip_delta=1.0):
        self.pid_config = pid_config
        self.sample_interval = int(sample_interval)  # default to be 10
        self.turn_controller = PID(K_P=1.0, K_I=0.75, K_D=0.4, n=10)
        self.speed_controller = PID(K_P=5.0, K_I=0.5, K_D=1.0, n=10)
        self.alpha = 0.5
        self.beta = 2.5
        self.min_aim_dis = 4.0
        self.max_aim_dis = 8.0
        self.max_throttle = max_throttle
        self.brake_speed = brake_speed
        self.brake_ratio = brake_ratio
        self.clip_delta = clip_delta
        self.desired_speed = None
        self.delta_angle = None

    def control_pid(self, output: np.ndarray, speed: float, tp):
        ''' Predicts vehicle control with a PID controller.
        Args:
            local_pos: the descrete waypoints of the planned trajectory
            speed: current speed (m/s)
        '''
        lon = "lon_reg_final"
        desired_speed = output[lon][0].numpy()
        # brake
        brake = desired_speed < self.brake_speed or (speed / desired_speed) > self.brake_ratio
        # throttle
        delta = np.clip(desired_speed - speed, 0.0, self.clip_delta)
        throttle = self.speed_controller.step(delta)
        throttle = np.clip(throttle, 0.0, self.max_throttle)
        throttle = throttle if not brake else 0.0

        ## angle
        lat = "lat_reg_final"
        local_pos = output[lat].numpy()
        aim_dist = np.clip(self.alpha * speed + self.beta, self.min_aim_dis, self.max_aim_dis)
        norms = np.linalg.norm(local_pos[:-1], axis=1)
        closest_index = np.abs(norms - aim_dist).argmin()
        aim = local_pos[closest_index]  # aim location 
        # steer
        angle = np.degrees(np.pi / 2 - np.arctan2(aim[1], aim[0])) / 90
        if speed < 0.01:
            # When we don't move we don't want the angle error to accumulate in the integral
            angle = 0.0
        if brake:
            angle = 0.0

        steer = self.turn_controller.step(angle)
        steer = np.clip(steer, -1.0, 1.0)

        self.desired_speed = desired_speed
        self.delta_angle = angle

        metadata = {
            'speed': float(speed.astype(np.float64)),
            'steer': float(steer),
            'throttle': float(throttle),
            'brake': float(brake),
            # 'wp_4': tuple(local_pos[3].astype(np.float64)),
            # 'wp_3': tuple(local_pos[2].astype(np.float64)),
            # 'wp_2': tuple(local_pos[1].astype(np.float64)),
            # 'wp_1': tuple(local_pos[0].astype(np.float64)),
            'aim': tuple(aim.astype(np.float64)),
            # 'target': tuple(target.astype(np.float64)),
            'desired_speed': float(desired_speed.astype(np.float64)),
            # 'angle': float(angle.astype(np.float64)),
            # 'angle_last': float(angle_last.astype(np.float64)),
            # 'angle_target': float(angle_target.astype(np.float64)),
            # 'angle_final': float(angle_final.astype(np.float64)),
            # 'delta': float(delta.astype(np.float64)),
        }
        return steer, throttle, brake, metadata