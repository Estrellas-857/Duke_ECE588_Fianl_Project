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

image_path = 'D:/archive/test/79-I-0877.png' # test image
img = cv2.imread(image_path)
processed_image = extract_license_plate(image_path)

###Extract license plate and find characters in the resulting images, completed - from traditional_LPR import extract_license_plate

###Segmenting the alphanumeric characters from the license plate.
def find_contours(dimensions, img):
     # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply binarization
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find all contours in the image
    cntrs, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Retrieve potential dimensions
    lower_width = dimensions[0]
    upper_width = dimensions[1]
    lower_height = dimensions[2]
    upper_height = dimensions[3]


    # Check largest 5 or  15 contours for license plate or character respectively
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

            char_copy = np.zeros((44,24))
            #extracting each character using the enclosing rectangle's coordinates.
            char = img[intY:intY+intHeight, intX:intX+intWidth]
            char = cv2.resize(char, (20, 40))

            # Make result formatted for classification: invert colors
            char = cv2.subtract(255, char)

            # Make sure the char is a grayscale image before copying to char_copy
            char_gray = cv2.cvtColor(char, cv2.COLOR_BGR2GRAY)

            # Resize the image to 24x44 with black border
            char_copy[2:42, 2:22] = char_gray
            char_copy[0:2, :] = 0
            char_copy[:, 0:2] = 0
            char_copy[42:44, :] = 0
            char_copy[:, 22:24] = 0

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
    # Set the number of rows and columns to display
    n_cols = len(images)
    n_rows = 1
    
    # Create a graph and a set of subgraphs
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 3))
    if n_cols == 1: # If there is only one character, convert axes to a list
        axes = [axes]
    
    for ax, img in zip(axes, images):
        ax.imshow(img, cmap='gray') 
        ax.axis('off') 

    #plt.show()

# Call the find_contours function to obtain the character image array
# The dimensions parameter is the estimated range of character sizes
dimensions = [15, 100, 100, 200]  #  width 15-100 pixels, height 100-200 pixels
char_images = find_contours(dimensions, processed_image)

#display_characters(char_images)

###create model
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5,5), input_shape=(28, 28, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.4))
model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=36, activation='softmax'))

train_datagen = ImageDataGenerator(rescale=1./255, width_shift_range=0.05, height_shift_range=0.05)

train_generator = train_datagen.flow_from_directory(
        'D:/archive/data/train',  # this is the target directory
        target_size=(28,28),  # all images will be resized to 28x28
        batch_size=1,
        class_mode='categorical',
        color_mode='grayscale')

validation_generator = train_datagen.flow_from_directory(
        'D:/archive/data/val',  # this is the target directory
        target_size=(28,28),  # all images will be resized to 28x28
        batch_size=1,
        class_mode='categorical',
        color_mode='grayscale')

# Compile model
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




### output
def fix_dimension(img): 
  new_img = np.zeros((28,28,3))
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
        img_ = cv2.resize(ch, (28,28))
        if len(img_.shape) == 3 and img_.shape[2] == 3:  # Check if it is a color image
            img_ = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)  # Convert to grayscale image
        
        #img = fix_dimension(img_)
        img = img_.reshape(1,28,28,1) #preparing image for the model
        predictions = model.predict(img, verbose=0)
        #y_ = model.predict_classes(img)[0] #predicting the class
        y_ = np.argmax(predictions, axis=1)[0]  # Prediction category
        character = dic[y_] #
        output.append(character) #storing the result in a list
        
    plate_number = ''.join(output)
    
    return plate_number

#print(show_results(char_images))


#plt.figure(figsize=(10,6))
#for i,ch in enumerate(char_images):
#     img = cv2.resize(ch, (28,28))
#     plt.subplot(3,4,i+1)
#     plt.imshow(img,cmap='gray')
#     plt.title(f'predicted: {show_results()[i]}')
#     plt.axis('off')
# plt.show()


