import numpy as np
np.set_printoptions(precision=3, floatmode='fixed', suppress=True)
import matplotlib.pyplot as plt

K_PATH = 1024
K_VELOCITY = 256


def interp1d_extrap(x, xp, fp):
    x = np.asarray(x, dtype=float)
    xp = np.asarray(xp, dtype=float)
    fp = np.asarray(fp, dtype=float)

    y = np.interp(x, xp, fp)  # 先做区间内插值 + 端点夹紧

    # 左侧线性外插
    m_left = (fp[1] - fp[0]) / (xp[1] - xp[0])
    left_mask = x < xp[0]
    y[left_mask] = fp[0] + m_left * (x[left_mask] - xp[0])

    # 右侧线性外插
    m_right = (fp[-1] - fp[-2]) / (xp[-1] - xp[-2])
    right_mask = x > xp[-1]
    y[right_mask] = fp[-1] + m_right * (x[right_mask] - xp[-1])

    return y

def interp_trajectory(path_cluster, velocity_cluster, interp_func=np.interp):
    num_velocity = velocity_cluster.shape[1]
    trajectory = np.zeros((K_PATH, K_VELOCITY, num_velocity, 2))
    trajectory_mask = np.ones((K_PATH, K_VELOCITY, num_velocity))

    for i in range(K_PATH):
        for j in range(K_VELOCITY):
            path = path_cluster[i]
            velocity = velocity_cluster[j]

            target_distance = np.cumsum(velocity * 0.5, axis=0)
            pad_path = np.concatenate([np.zeros((1, 2)), path], axis=0)
            distance = np.linalg.norm(pad_path[1:, :2] - pad_path[:-1, :2], axis=-1).cumsum(axis=0)
            distance = np.concatenate([np.zeros((1,)), distance], axis=0)
            interp_trajectory = np.array(
                [
                    interp_func(target_distance, distance, pad_path[:, 0]),
                    interp_func(target_distance, distance, pad_path[:, 1]),
                ]
            ).T
            trajectory[i, j] = interp_trajectory

            max_dist = distance[-1]
            valid = target_distance <= max_dist
            trajectory_mask[i, j, ~valid] = 0.0

    return trajectory, trajectory_mask

path_cluster = np.load("data/kmeans/path_1m_pts_15_1024_b2d_new_ego.npy")
velocity_cluster = np.load("data/kmeans/vel_seq_K256_t30.npy")
trajectory, trajectory_mask = interp_trajectory(path_cluster, velocity_cluster, interp_func=interp1d_extrap)

for i in range(K_PATH):
    for j in range(K_VELOCITY):
        plt.plot(trajectory[i, j, :, 0], trajectory[i, j, :,1])
plt.savefig(f'vis/kmeans/trajectory_{K_PATH}_{K_VELOCITY}', bbox_inches='tight')
plt.close()

np.savez(f'data/kmeans/trajectory_{K_PATH}_{K_VELOCITY}.npz', trajectory=trajectory, trajectory_mask=trajectory_mask)
