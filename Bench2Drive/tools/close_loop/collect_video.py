import cv2
import os

def images_to_video(image_folder, output_video, downsample=4, fps=5):
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg") or img.endswith(".png")]
    images.sort()  # 确保图片按名称排序

    # 检查是否有图片
    if not images:
        print("没有找到任何图片")
        return

    # 获取第一张图片以确定视频的宽度和高度
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape
    frame = cv2.resize(frame, (width//downsample, height //
                    downsample), interpolation=cv2.INTER_AREA)

    # 定义编码器（-1会弹出选择框），创建VideoWriter对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 对于mp4格式
    video = cv2.VideoWriter(output_video, fourcc, fps, (width//downsample, height//downsample))

    for image in images:
        print(image)
        image_path = os.path.join(image_folder, image)
        frame = cv2.imread(image_path)
        frame = cv2.resize(frame, (width//downsample, height //
                    downsample), interpolation=cv2.INTER_AREA)
        video.write(frame)

    # 释放资源
    cv2.destroyAllWindows()
    video.release()


result_dir = "close_loop_log"
save_dir = os.path.join(result_dir, "save")
video_dir = os.path.join(result_dir, "videos")
os.makedirs(video_dir, exist_ok=True)

files = os.listdir(save_dir)
for file in files:
    video_path =  os.path.join(save_dir, file, "video.mp4")
    # import ipdb; ipdb.set_trace()
    if not os.path.exists(video_path):
        images_to_video(os.path.join(save_dir, file, "combine"), video_path)
    os.system(f"cp {video_path} {video_dir}/{file}.mp4 ")

