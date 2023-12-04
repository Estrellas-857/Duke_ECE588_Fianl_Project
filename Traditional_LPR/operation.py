import os
import time
from traditional_LPR import extract_license_plate, read_license_plate

# 路径设置
image_folder = 'D:/588way2/archive/test'  # 替换为图片文件夹路径
max_images_to_process = 1000  # 设置要处理的最大图片数量

# 统计变量
total_images = 0
correctly_identified = 0
error_cases = []  # 用于记录识别错误的情况

# 记录时间开始
start_time = time.time()

# 获取所有图片文件名，并限制处理数量
image_files = [img for img in os.listdir(image_folder) if img.endswith(('.png', '.jpg', '.jpeg'))]
image_files = image_files[:max_images_to_process]  # 限制处理数量

# 遍历文件夹内的所有图片文件
for image_name in image_files:
    total_images += 1
    image_path = os.path.join(image_folder, image_name)

    # 定位并识别车牌
    license_plate_image = extract_license_plate(image_path)
    identified_text = read_license_plate(license_plate_image)

    # 从文件名中提取正确答案
    correct_answer = image_name.split('.')[0].replace('-', '')

    # 比较结果并处理可能的错误
    if identified_text:
        identified_text_cleaned = identified_text.strip().replace(" ", "").upper()
        if identified_text_cleaned == correct_answer.upper():
            correctly_identified += 1
        else:
            error_cases.append((correct_answer, identified_text_cleaned))
    else:
        error_cases.append((correct_answer, 'None'))

# 打印错误案例，每个案例一行
for case in error_cases:
    print(f'Correct: {case[0]}, Identified: {case[1]}')

# 计算运行时间
end_time = time.time()
total_time = end_time - start_time

# 计算正确率
accuracy = (correctly_identified / total_images) * 100

# 输出统计结果
#print(error_cases)
print(f'Total images processed: {total_images}')
print(f'Correctly identified: {correctly_identified}')
print(f'Accuracy: {accuracy:.2f}%')
print(f'Total runtime: {total_time:.2f} seconds')
