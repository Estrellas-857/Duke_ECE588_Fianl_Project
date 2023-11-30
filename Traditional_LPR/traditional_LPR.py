import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract
from PIL import Image
# 指定 Tesseract-OCR 安装路径
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'


def extract_license_plate(image_path):
    # 读取图像
    image = cv2.imread(image_path)
    # 检查图像是否正确加载
    if image is None:
        print("Error: Unable to load image at path:", image_path)
        return

    # 如果图像加载成功，则进行后续处理
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 显示灰度图
    #plt.figure(figsize=(10, 6))
    #plt.subplot(131), plt.imshow(gray, cmap='gray'), plt.title('Gray Image')

    # 使用 Sobel 算子进行水平和垂直方向的边缘检测
    sobel_x = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)  # 水平方向
    sobel_y = cv2.Sobel(gray, cv2.CV_8U, 0, 1, ksize=3)  # 垂直方向

    # 结合水平和垂直方向的边缘检测结果
    sobel_combined = cv2.bitwise_or(sobel_x, sobel_y)

    # 显示 Sobel 边缘检测结果
    #plt.subplot(132), plt.imshow(sobel_combined, cmap='gray'), plt.title('Sobel Edge Detection')

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    ret, threshold = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)

    # 二值化
    #ret, threshold = cv2.threshold(sobel_combined, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)

    # 显示二值化结果
    #plt.subplot(133), plt.imshow(threshold, cmap='gray'), plt.title('Thresholding Result')
    #plt.show()

    # 对图像进行腐蚀和膨胀操作
    element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 6))
    element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 2))

    dilation = cv2.dilate(threshold, element2, iterations=1)
    erosion = cv2.erode(dilation, element1, iterations=1)
    #dilation2 = cv2.dilate(erosion, element2, iterations=3)

    # 显示膨胀腐蚀结果
    #plt.figure(figsize=(10, 6))
    #plt.subplot(131), plt.imshow(dilation, cmap='gray'), plt.title('Dilation Result')
    #plt.subplot(132), plt.imshow(erosion, cmap='gray'), plt.title('Erosion Result')
    #plt.subplot(133), plt.imshow(dilation2, cmap='gray'), plt.title('Final Dilation Result')
    #plt.show()

    # 查找车牌区域
    license_plate = None
    contours, hierarchy = cv2.findContours(erosion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h
        if 2 < aspect_ratio < 5.5:  # 车牌正常情况下长宽比在2到5.5之间
            license_plate = image[y:y + h, x:x + w]
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            break  # 假设只有一个车牌，找到后就退出循环

    # 显示结果
    #plt.figure(figsize=(10, 6))
    #plt.subplot(121), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('Original Image')
    #if license_plate is not None:
        #plt.subplot(122), plt.imshow(cv2.cvtColor(license_plate, cv2.COLOR_BGR2RGB)), plt.title('License Plate')
    #else:
        #plt.subplot(122), plt.imshow(np.zeros((50, 150, 3), dtype=np.uint8)), plt.title('License Plate Not Found')
    #plt.show()

    # 返回识别到的车牌图像
    return license_plate

def read_license_plate(license_plate_img):
    if license_plate_img is not None:
        # 将OpenCV图像转换为PIL图像格式
        license_plate_img_pil = Image.fromarray(cv2.cvtColor(license_plate_img, cv2.COLOR_BGR2RGB))

        # Tesseract配置：使用白名单来限制字符集
        config = '-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 8'

        # 使用Tesseract进行OCR
        text = pytesseract.image_to_string(license_plate_img_pil, config=config)

        # 可选：使用正则表达式进行后处理
        # import re
        # pattern = re.compile("[A-Z0-9]+")
        # matches = pattern.findall(text)
        # if matches:
        #     text = " ".join(matches)

        #print(text)  # 输出识别到的文本
        return text
    else:
        print("No license plate found")




if __name__ == "__main__":
    # 替换为车辆图片路径
    image_path = 'D:/archive/test/1/80-ZL-029.png'#for test
    license_plate_img = extract_license_plate(image_path)
    read_license_plate(license_plate_img)

