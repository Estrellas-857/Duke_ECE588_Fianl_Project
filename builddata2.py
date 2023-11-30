import cv2
import os
import numpy as np

# 指定原始图像文件夹和目标文件夹路径
original_folder_path = 'D:/588way2/newdata/oldtrain'
processed_folder_path = 'D:/588way2/newdata/train'

# 如果目标文件夹不存在，则创建它
if not os.path.exists(processed_folder_path):
    os.makedirs(processed_folder_path)

def process_and_crop_image(image_path, crop_size=(110, 120), threshold=127):
    # 读取图像
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 颜色二值化
    _, binary_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)

    # 颜色翻转
    inverted_image = cv2.bitwise_not(binary_image)

    # 确保图像尺寸足够大
    h, w = inverted_image.shape
    padding_color = 0  # 黑色填充，因为是灰度图像
    if h < crop_size[1]:
        padding_h = crop_size[1] - h
        inverted_image = cv2.copyMakeBorder(inverted_image, 0, padding_h, 0, 0, cv2.BORDER_CONSTANT, value=padding_color)
    if w < crop_size[0]:
        padding_w = crop_size[0] - w
        inverted_image = cv2.copyMakeBorder(inverted_image, 0, 0, 0, padding_w, cv2.BORDER_CONSTANT, value=padding_color)

    # 裁剪图像
    cropped_image = inverted_image[:crop_size[1], :crop_size[0]]
    return cropped_image



    # # 再次确认尺寸并裁剪
    # h, w = image.shape[:2]
    # cropped_image = image[:crop_size[1], :crop_size[0]]
    # return cropped_image

def center_and_pad_image(image_path, target_size=(120, 120), padding_color=0):
    # 读取图像
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 计算原始图像的中心点
    h, w = image.shape
    center_y, center_x = h // 2, w // 2

    # 创建120x120的黑色背景
    new_image = np.zeros(target_size, dtype=np.uint8)

    # 计算将原图放置在新图像中心的偏移量
    offset_y = max(target_size[0] // 2 - center_y, 0)
    offset_x = max(target_size[1] // 2 - center_x, 0)

    # 计算在原图和新图中应该复制的区域
    copy_region_h = min(target_size[0], h)
    copy_region_w = min(target_size[1], w)

    # 将原图复制到新图的中心
    new_image[offset_y:offset_y + copy_region_h, offset_x:offset_x + copy_region_w] = image[center_y - copy_region_h // 2:center_y + (copy_region_h+1) // 2, center_x - copy_region_w // 2:center_x + (copy_region_w+1) // 2]

    return new_image


def extract_label_from_filename(filename):
    # 假设文件名格式为 "图片名.jpg"
    label = filename[0]  # 或者适用于您的文件名格式的其他逻辑
    return label



# 遍历原始文件夹中的所有图像
for filename in os.listdir(original_folder_path):
    if filename.endswith(('.png', '.jpg', '.jpeg')):
        # 构建完整的文件路径
        original_image_path = os.path.join(original_folder_path, filename)

        # 处理图像
        processed_image = process_and_crop_image(original_image_path, crop_size=(120, 120))

        # 构建目标图像的路径
        processed_image_path = os.path.join(processed_folder_path, filename)

        # 保存处理后的图像
        cv2.imwrite(processed_image_path, processed_image)
        
labels = {filename: extract_label_from_filename(filename) for filename in os.listdir(processed_folder_path) if filename.endswith(('.png', '.jpg', '.jpeg'))}

print((labels))