import os
from datagenerator import CustomDataGenerator

def get_data(data_dir):
    filenames = os.listdir(data_dir)
    full_paths = [os.path.join(data_dir, f).replace('\\', '/') for f in filenames if f.endswith('.png') or f.endswith('.jpg')]
    return full_paths

# Specify data directory
data_dir = "D:/archive/train"

# Get dataset
image_paths = get_data(data_dir)
print(image_paths[:10])

# Define the parameters required by the data generator
batch_size = 50  
img_width = 1025  # image width
img_height = 218  # image height
downsample_factor = 2  # Adjust according to CNN structure
max_text_length = 7  # Maximum character length of license plate

# Create a data generator instance
train_gen = CustomDataGenerator(image_paths, batch_size, img_width, img_height, downsample_factor, max_text_length)

# Next is model definition and training...
print('start model define and training')
