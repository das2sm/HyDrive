import cv2
import os



def images_to_video(image_folder, output_video, fps=4):
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

    # 定义编码器（-1会弹出选择框），创建VideoWriter对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 对于mp4格式
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    for image in images:
        print(image)
        image_path = os.path.join(image_folder, image)
        frame = cv2.imread(image_path)
        video.write(frame)

    # 释放资源
    cv2.destroyAllWindows()
    video.release()

# 使用函数
images_to_video('vis/col', 'output_video.mp4')