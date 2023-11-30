# 导入必要的库
import os
from datagenerator import CustomDataGenerator

def get_data(data_dir):
    filenames = os.listdir(data_dir)
    full_paths = [os.path.join(data_dir, f).replace('\\', '/') for f in filenames if f.endswith('.png') or f.endswith('.jpg')]
    return full_paths

# 指定数据目录
data_dir = "D:/archive/train"

# 获取数据集
image_paths = get_data(data_dir)
print(image_paths[:10])

# 定义数据生成器所需的参数
batch_size = 50  # 或您选择的其他数值
img_width = 1025  # 您图像的宽度
img_height = 218  # 您图像的高度
downsample_factor = 2  # 根据您的CNN结构调整
max_text_length = 7  # 车牌的最大字符长度

# 创建数据生成器实例
train_gen = CustomDataGenerator(image_paths, batch_size, img_width, img_height, downsample_factor, max_text_length)

# 接下来是模型定义和训练...
print('start model define and training')