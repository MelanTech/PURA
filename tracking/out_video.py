import cv2
import os
import re

# 文件夹路径和输出视频文件名
image_folder = './dog1_tta'
video_file = 'dog1_tta.avi'


# 获取文件夹中的所有图片文件名，并按数字排序
def sort_naturally(l):
    # 自定义排序函数，处理数字排序
    def alphanum_key(s):
        return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

    return sorted(l, key=alphanum_key)


images = [img for img in os.listdir(image_folder) if img.endswith(".jpg") or img.endswith(".png")]
images = sort_naturally(images)

# 读取第一张图片以获取视频的宽度和高度
first_image = cv2.imread(os.path.join(image_folder, images[0]))
height, width, _ = first_image.shape

# 创建视频写入对象
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video = cv2.VideoWriter(video_file, fourcc, 30, (width, height))

# 将每张图片添加到视频中
for image in images:
    print(image)
    img_path = os.path.join(image_folder, image)
    img = cv2.imread(img_path)
    video.write(img)

# 释放视频写入对象
video.release()
cv2.destroyAllWindows()
