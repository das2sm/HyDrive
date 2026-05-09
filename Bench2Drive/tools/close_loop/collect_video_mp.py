import cv2
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

# ========== 纯函数，在子进程里执行 ==========
def _process_one_folder(sub_folder, save_root, video_root, downsample=4, fps=5):
    """
    sub_folder: 子目录名（不含路径）
    save_root : 所有子目录的公共父目录
    video_root: 最终视频统一拷贝到的目录
    """
    src_dir   = os.path.join(save_root, sub_folder, "combine")
    dst_video = os.path.join(save_root, sub_folder, "video.mp4")
    final_mp4 = os.path.join(video_root, f"{sub_folder}.mp4")

    # 若目标已存在，直接跳过
    if os.path.exists(dst_video):
        os.system(f'cp "{dst_video}" "{final_mp4}"')
        return f"{sub_folder}: already done"

    # 1. 收集图片
    images = [i for i in os.listdir(src_dir) if i.lower().endswith((".jpg", ".png"))]
    if not images:
        return f"{sub_folder}: no images"
    images.sort()

    # 2. 获取第一帧尺寸
    first = cv2.imread(os.path.join(src_dir, images[0]))
    h, w  = first.shape[:2]
    w_new, h_new = w // downsample, h // downsample

    # 3. 创建 writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(dst_video, fourcc, fps, (w_new, h_new))
    if not writer.isOpened():
        return f"{sub_folder}: VideoWriter open failed"

    # 4. 逐帧写入
    for img_name in images:
        frame = cv2.imread(os.path.join(src_dir, img_name))
        frame = cv2.resize(frame, (w_new, h_new), interpolation=cv2.INTER_AREA)
        writer.write(frame)

    writer.release()
    # 拷贝到统一目录
    os.system(f'cp "{dst_video}" "{final_mp4}"')
    return f"{sub_folder}: finished"


# ========== 主入口 ==========
def images_to_video_parallel(result_dir="close_loop_log",
                             downsample=4,
                             fps=5,
                             max_workers=None):
    save_dir  = os.path.join(result_dir, "save")
    video_dir = os.path.join(result_dir, "videos")
    os.makedirs(video_dir, exist_ok=True)

    sub_dirs = [d for d in os.listdir(save_dir)
                if os.path.isdir(os.path.join(save_dir, d, "combine"))]

    # 用 partial 把固定参数绑进去，子进程只接收子目录名
    worker = partial(_process_one_folder,
                     save_root=os.path.abspath(save_dir),
                     video_root=os.path.abspath(video_dir),
                     downsample=downsample,
                     fps=fps)

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(worker, d) for d in sub_dirs]
        for f in as_completed(futures):
            print(f.result())        # 实时打印子进程返回信息


if __name__ == "__main__":
    images_to_video_parallel(max_workers=10)