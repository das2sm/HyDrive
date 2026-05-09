import cv2, pathlib, glob

img_dir  = pathlib.Path('.')                 # 当前路径
jpg_list = sorted(glob.glob('all*.jpg'))    # 按文件名自然排序
if not jpg_list:
    raise FileNotFoundError('找不到 all*.jpg')

# 读取第一张拿到尺寸
frame0 = cv2.imread(str(jpg_list[0]))
h, w, _ = frame0.shape

out = cv2.VideoWriter('video.mp4',
                      cv2.VideoWriter_fourcc(*'mp4v'),
                      10,           # fps
                      (w, h))

for p in jpg_list:
    out.write(cv2.imread(str(p)))

out.release()
print('video.mp4 已生成')