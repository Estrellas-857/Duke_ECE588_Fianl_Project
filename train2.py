#importing openCV
import cv2

#importing numpy
import numpy as np

#importing pandas to read the CSV file containing our data
import pandas as pd

#importing keras and sub-libraries
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten, MaxPool2D
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras import backend as K
from keras.utils import to_categorical
#from keras.utils import np_utils
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from traditional_LPR import extract_license_plate

from keras.preprocessing.image import ImageDataGenerator

import time
import datetime
import tensorflow as tf
import os

image_path = 'D:/archive/test/79-I-0877.png'
img = cv2.imread(image_path)
processed_image = extract_license_plate(image_path)
###提取license plate，已完成

###find characters in the resulting images 已完成

###Segmenting the alphanumeric characters from the license plate.
def find_contours(dimensions, img):
     # 将图像转换为灰度图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 应用二值化处理
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 显示处理后的二值化图像
    cv2.imshow("Binary Image", binary)
    cv2.waitKey(0)  # 等待按键后关闭窗口
    cv2.destroyAllWindows()

    # Find all contours in the image
    cntrs, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Retrieve potential dimensions
    lower_width = dimensions[0]
    upper_width = dimensions[1]
    lower_height = dimensions[2]
    upper_height = dimensions[3]


    # Check largest 7 contours for license plate or character respectively
    cntrs = sorted(cntrs, key=cv2.contourArea, reverse=True)[:15]

    x_cntr_list = []
    target_contours = []
    img_res = []
    for cntr in cntrs :
        #detects contour in binary image and returns the coordinates of rectangle enclosing it
        intX, intY, intWidth, intHeight = cv2.boundingRect(cntr)

        #checking the dimensions of the contour to filter out the characters by contour's size
        if intWidth > lower_width and intWidth < upper_width and intHeight > lower_height and intHeight < upper_height :
            x_cntr_list.append(intX) #stores the x coordinate of the character's contour, to used later for indexing the contours

            # 提取每个字符
            char = img[intY:intY + intHeight, intX:intX + intWidth]

            # 如果图像是彩色的，转换为灰度图像
            if len(char.shape) == 3:
                char = cv2.cvtColor(char, cv2.COLOR_BGR2GRAY)

                # 颜色翻转
            char = cv2.bitwise_not(char)

            # 保持原始宽高比缩放字符
            scale_ratio = 100.0 / max(char.shape[:2])
            char_resized = cv2.resize(char, (int(char.shape[1] * scale_ratio), int(char.shape[0] * scale_ratio)), interpolation=cv2.INTER_AREA)

            char_copy = np.zeros((120,120), dtype=np.uint8)

            # 计算字符放置位置
            x_offset = (120 - char_resized.shape[1]) // 2
            y_offset = (120 - char_resized.shape[0]) // 2

            # 在背景中心放置字符
            char_copy[y_offset:y_offset + char_resized.shape[0], x_offset:x_offset + char_resized.shape[1]] = char_resized

            # #extracting each character using the enclosing rectangle's coordinates.
            # char = img[intY:intY+intHeight, intX:intX+intWidth]
            # char = cv2.resize(char, (100, 100))

            # # Make result formatted for classification: invert colors
            # char = cv2.subtract(255, char)

            # # 在复制到 char_copy 之前，确保 char 是灰度图像
            # char_gray = cv2.cvtColor(char, cv2.COLOR_BGR2GRAY)

            # # Resize the image to 24x44 with black border
            # char_copy[2:42, 2:22] = char_gray
            # char_copy[0:2, :] = 0
            # char_copy[:, 0:2] = 0
            # char_copy[42:44, :] = 0
            # char_copy[:, 22:24] = 0

            img_res.append(char_copy) #List that stores the character's binary image (unsorted)

    #Return characters on ascending order with respect to the x-coordinate (most-left character first)

    #arbitrary function that stores sorted list of character indeces
    indices = sorted(range(len(x_cntr_list)), key=lambda k: x_cntr_list[k])
    img_res_copy = []
    for idx in indices:
        img_res_copy.append(img_res[idx])# stores character images according to their index
    img_res = np.array(img_res_copy)

    return img_res

def display_characters(images):
    # 设定显示的行数和列数
    n_cols = len(images)
    n_rows = 1
    
    # 创建一个图形和一组子图
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 3))
    if n_cols == 1: # 如果只有一个字符，则将 axes 转换为列表
        axes = [axes]
    
    for ax, img in zip(axes, images):
        ax.imshow(img, cmap='gray') # 显示字符图像
        ax.axis('off') # 关闭坐标轴显示

    plt.show()

# 调用 find_contours 函数获取字符图像数组
# 假设您已经有了车牌的二值图像 plate_binary
# dimensions 参数是字符尺寸的预估范围，需要根据实际情况调整
dimensions = [15, 150, 100, 200]  # 示例：宽度15-100像素，高度100-200像素
char_images = find_contours(dimensions, processed_image)

# 使用 display_characters 函数显示字符
display_characters(char_images)

for img in char_images:
    print(img.shape)  # 输出每个字符图像的像素尺寸

###create model
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5,5), input_shape=(120, 120, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.4))
model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=36, activation='softmax'))

train_datagen = ImageDataGenerator(rescale=1./255, width_shift_range=0.10, height_shift_range=0.10)

train_generator = train_datagen.flow_from_directory(
        'D:/588way2/newdata/train',  # this is the target directory
        target_size=(120,120),  # all images will be resized to 120*120
        batch_size=1,
        class_mode='categorical',
        color_mode='grayscale')

validation_generator = train_datagen.flow_from_directory(
        'D:/588way2/newdata/val',  # this is the target directory
        target_size=(120,120),  # all images will be resized to 120*120
        batch_size=1,
        class_mode='categorical',
        color_mode='grayscale')

# 编译模型
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

class stop_training_callback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('val_accuracy', 0) > 0.992):
      self.model.stop_training = True
      
log_dir="D:/archive/new/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

batch_size = 1
callbacks = [tensorboard_callback, stop_training_callback()]
model.fit_generator(train_generator,
      steps_per_epoch = train_generator.samples // batch_size,
      validation_data = validation_generator, 
      validation_steps = validation_generator.samples // batch_size,
      epochs = 80, callbacks=callbacks)




###output
def fix_dimension(img): 
  new_img = np.zeros((120,120,3))
  for i in range(3):
    new_img[:,:,i] = img
  return new_img
  
def show_results(char_images):
    dic = {}
    characters = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    for i,c in enumerate(characters):
        dic[i] = c

    output = []
    for i,ch in enumerate(char_images): #iterating over the characters
        img_ = cv2.resize(ch, (120,120))
        if len(img_.shape) == 3 and img_.shape[2] == 3:  # 检查是否为彩色图像
            img_ = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)  # 转换为灰度图像
        
        #img = fix_dimension(img_)
        img = img_.reshape(1,120,120,1) #preparing image for the model
        predictions = model.predict(img, verbose=0)
        #y_ = model.predict_classes(img)[0] #predicting the class
        y_ = np.argmax(predictions, axis=1)[0]  # 预测类别
        character = dic[y_] #
        output.append(character) #storing the result in a list
        
    plate_number = ''.join(output)
    
    return plate_number

predicted_plate = show_results(char_images)
print(show_results(char_images))


plt.figure(figsize=(10,6))
for i,ch in enumerate(char_images):
     img = cv2.resize(ch, (120,120))
     plt.subplot(3,4,i+1)
     plt.imshow(img,cmap='gray')
     plt.title(f'predicted: {predicted_plate[i]}')
     plt.axis('off')
plt.show()


def show_dictionary(dic):
    for key, value in dic.items():
        print(f"Key: {key}, Value: {value}")

# dic = {}
# characters = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
# for i, c in enumerate(characters):
#     dic[i] = c

# show_dictionary(dic)

# x_batch, y_batch = next(train_generator)
# print(f"Batch shape: {x_batch.shape}")
# print(f"Labels: {y_batch}")

# # 可视化第一个图像和其标签
# plt.imshow(x_batch[0].reshape(120, 120), cmap='gray')
# plt.title(f"Label: {y_batch[0]}")
# plt.show()

# output_layer = model.layers[-1]
# print(f"Output layer type: {type(output_layer)}")
# print(f"Number of neurons in the output layer: {output_layer.output_shape[1]}")

# # 预期的类别数量
# expected_num_classes = len(characters)
# print(f"Expected number of classes: {expected_num_classes}")
