from keras.utils import Sequence
import numpy as np
import cv2
import os

class CustomDataGenerator(Sequence):
    def __init__(self, image_paths, batch_size, img_width, img_height, downsample_factor, max_text_length):
        self.image_paths = image_paths  
        self.batch_size = batch_size
        self.img_width = img_width
        self.img_height = img_height
        self.downsample_factor = downsample_factor
        self.max_text_length = max_text_length
        self.char_list = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'

    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def __getitem__(self, idx):
        batch_start = idx * self.batch_size
        batch_end = min((idx + 1) * self.batch_size, len(self.image_paths))
        batch_size = batch_end - batch_start
        batch_paths = self.image_paths[batch_start:batch_end]




        #print(f"The image path of the current batch: {batch_paths}") 

        x = np.zeros((batch_size, self.img_height, self.img_width, 3), dtype=np.float32)
        y = np.ones([batch_size, self.max_text_length]) * -1
        #y = np.zeros((batch_size, self.max_text_length, self.num_classes), dtype=np.float32)  # Use one-hot encoding
        input_length = np.ones((batch_size, 1)) * (self.img_width // self.downsample_factor )
        label_length = np.zeros((batch_size, 1))

        for i, path in enumerate(batch_paths):
            #print(f"currently processed image path: {path}")  

            img = cv2.imread(path)
            if img is None:
                print(f"Warning: unable to load the image {path}")
                continue

            #print(f"image size: {img.shape}")  # Print image size to confirm image has been loaded

            img = cv2.resize(img, (self.img_width, self.img_height))
            img = img.astype(np.float32) / 255.0
            x[i] = img

            label = os.path.basename(path).split('.')[0].replace('-', '')
            #print(f"label: {label}")  # Print the corresponding label

            label_length[i] = len(label)
            y[i, :len(label)] = [self.char_list.index(char) for char in label if char in self.char_list]

        inputs = {'image_input': x, 'labels': y, 'input_length': input_length, 'label_length': label_length}
        outputs = {'ctc': np.zeros([batch_size])}

        return inputs, outputs

