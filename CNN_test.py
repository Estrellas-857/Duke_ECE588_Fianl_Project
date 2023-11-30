import os
import time
from traditional_LPR import extract_license_plate
from train2 import find_contours, show_results, dimensions
import cv2

###for test only
folder_path = 'D:/archive/test/'

# 获取文件夹中的所有图片文件
image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

# 构建正确答案的字典
correct_answers = {}
for image_file in image_files:
    # 移除文件后缀并替换'-'字符
    plate_number = os.path.splitext(image_file)[0].replace('-', '')
    correct_answers[image_file] = plate_number


# 获取文件夹中的所有图片文件
image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

correct_count = 0
total_time = 0

# 遍历并处理每个图像
for image_file in image_files:
    img_path = os.path.join(folder_path, image_file)
    img = cv2.imread(img_path)

    start_time = time.time()
    processed_image = extract_license_plate(img_path)
    char_images = find_contours(dimensions, processed_image)
    predicted_plate = show_results(char_images)
    total_time += time.time() - start_time

    correct_plate = correct_answers.get(image_file, '')
    # 检查预测结果是否正确
    if predicted_plate == correct_plate:
        correct_count += 1
    else:
        print(f"Image: {image_file}, Predicted Plate: {predicted_plate}, Correct Plate: {correct_plate}")

# 计算正确率和平均耗时
accuracy = correct_count / len(image_files)
average_time = total_time / len(image_files)

print(f"Accuracy: {accuracy*100:.2f}%, Average Time per Image: {average_time:.2f} seconds")
print(f"{total_time:.2f} seconds")