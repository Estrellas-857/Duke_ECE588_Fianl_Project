from datagenerator import CustomDataGenerator
import tensorflow as tf
from keras.layers import Input, Conv2D, MaxPooling2D, Reshape, Dense, LSTM, Lambda
from keras.models import Model
from keras.backend import ctc_batch_cost
from imagepath import get_data
import os
import time
import numpy as np

# 函数定义：CTC损失计算
def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return ctc_batch_cost(labels, y_pred, input_length, label_length)

# 输入层定义
input_img = Input(shape=(1025, 218, 3), name='image_input')
labels = Input(name='labels', shape=[None], dtype='float32')
input_length = Input(name='input_length', shape=[1], dtype='int64')
label_length = Input(name='label_length', shape=[1], dtype='int64')

# CNN层
x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2))(x)
# ... 添加更多的卷积和池化层 ...

# RNN层
# 根据前面的层调整
new_shape = (512, 16 * 109)  # 这个尺寸需要根据实际的CNN层输出来计算
x = Reshape(target_shape=new_shape)(x)
x = Dense(64, activation='relu')(x)
x = LSTM(128, return_sequences=True)(x)

# 输出层
x = Dense(37, activation='softmax', name='output')(x)

# CTC损失函数
loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([x, labels, input_length, label_length])

# 定义最终模型
model = Model(inputs=[input_img, labels, input_length, label_length], outputs=loss_out)

# 编译模型
model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adam')


# 打印模型结构
model.summary()

train_image_paths = get_data("D:/archive/train")
val_image_paths = get_data("D:/archive/test")

#print("训练图像路径: ", train_image_paths[:10])  # 打印前10个训练图像路径进行检查
#print("验证图像路径: ", val_image_paths[:10])    # 打印前10个验证图像路径进行检查

# 训练模型
# 定义模型参数
batch_size = 50  # 选择合适的批次大小
img_width = 1025  # 图像宽度
img_height = 218  # 图像高度
downsample_factor = 2  # 根据CNN结构调整
max_text_length = 7  # 车牌最大字符长度

# 创建数据生成器实例
train_gen = CustomDataGenerator(train_image_paths, batch_size, img_width, img_height, downsample_factor, max_text_length)
val_gen = CustomDataGenerator(val_image_paths, batch_size, img_width, img_height, downsample_factor, max_text_length)

model_path = 'D:/archive/model'
# 使用数据生成器训练模型
if not os.path.exists(model_path):
    # 训练模型
    history = model.fit(
        train_gen,
        epochs=1,  # 选择迭代次数
        validation_data=val_gen,
        steps_per_epoch=200,  # 为训练样本数除以批次大小
        validation_steps=20  # 为验证样本数除以批次大小
    )
    model.save(model_path)
    print('Model trained and saved!')
else:
    print('Model already exists, loading...')
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={'ctc_lambda_func': ctc_lambda_func}
    )

# 加载测试数据
def decode_predictions(preds, char_list):
    decoded_texts = []
    for pred in preds:
        # 使用 argmax 获取每个时间步最可能的字符
        pred_text = ''
        for t in range(pred.shape[0]):
            pred_char = np.argmax(pred[t])
            if pred_char != len(char_list):  # 空白标签不添加
                pred_text += char_list[pred_char]
        decoded_texts.append(pred_text)
    return decoded_texts

# 进行预测
predictions = model.predict(val_gen)

# 解码预测结果
decoded_predictions = decode_predictions(predictions, char_list='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')

start_time = time.time()
correct_predictions = 0
total_images = len(val_image_paths)

for i, path in enumerate(val_image_paths):
    true_label = os.path.basename(path).split('.')[0].replace('-', '')
    predicted_label = decoded_predictions[i]

    print("真实标签:", true_label)
    print("预测标签:", predicted_label)

    if true_label == predicted_label:
        correct_predictions += 1

accuracy = correct_predictions / total_images
end_time = time.time()
total_time = end_time - start_time

print(f"准确率: {accuracy * 100:.2f}%")
print(f"总运行时间: {total_time:.2f}秒")
print(f"处理的图片数量: {total_images}")