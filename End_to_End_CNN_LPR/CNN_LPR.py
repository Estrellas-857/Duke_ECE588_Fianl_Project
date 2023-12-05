from datagenerator import CustomDataGenerator
import tensorflow as tf
from keras.layers import Input, Conv2D, MaxPooling2D, Reshape, Dense, LSTM, Lambda
from keras.models import Model
from keras.backend import ctc_batch_cost
from imagepath import get_data
import os
import time
import numpy as np

# Function definition: CTC loss calculation
def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return ctc_batch_cost(labels, y_pred, input_length, label_length)

# Input layer definition
input_img = Input(shape=(1025, 218, 3), name='image_input')
labels = Input(name='labels', shape=[None], dtype='float32')
input_length = Input(name='input_length', shape=[1], dtype='int64')
label_length = Input(name='label_length', shape=[1], dtype='int64')

# CNN layer
x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2))(x)
# ... add more convolution and pooling layers ...

# RNN layer
new_shape = (512, 16 * 109)
x = Reshape(target_shape=new_shape)(x)
x = Dense(64, activation='relu')(x)
x = LSTM(128, return_sequences=True)(x)

# output layer
x = Dense(37, activation='softmax', name='output')(x)

# CTC loss function
loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([x, labels, input_length, label_length])

# Define the final model
model = Model(inputs=[input_img, labels, input_length, label_length], outputs=loss_out)

# Compile model
model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adam')


# Print model structure
model.summary()

train_image_paths = get_data("D:/archive/train")
val_image_paths = get_data("D:/archive/test")

#print("training image path: ", train_image_paths[:10])  # Print the first 10 training image paths for inspection
#print("validation image path: ", val_image_paths[:10])    

#Train model
# Define model parameters
batch_size = 50  
img_width = 1025  # Image width
img_height = 218  # Image height
downsample_factor = 2  # Adjust according to CNN structure
max_text_length = 7  # Maximum character length of license plate

# Create a data generator instance
train_gen = CustomDataGenerator(train_image_paths, batch_size, img_width, img_height, downsample_factor, max_text_length)
val_gen = CustomDataGenerator(val_image_paths, batch_size, img_width, img_height, downsample_factor, max_text_length)

model_path = 'D:/archive/model'
# Train a model using a data generator
if not os.path.exists(model_path):
    # train model
    history = model.fit(
        train_gen,
        epochs=1,  # iteration number
        validation_data=val_gen,
        steps_per_epoch=200,  # is the number of training samples divided by the batch size
        validation_steps=20  # is the number of validation samples divided by the batch size
    )
    model.save(model_path)
    print('Model trained and saved!')
else:
    print('Model already exists, loading...')
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={'ctc_lambda_func': ctc_lambda_func}
    )

# Load test data
def decode_predictions(preds, char_list):
    decoded_texts = []
    for pred in preds:
        # Use argmax to get the most likely character at each time step
        pred_text = ''
        for t in range(pred.shape[0]):
            pred_char = np.argmax(pred[t])
            if pred_char != len(char_list):  # Blank tags are not added
                pred_text += char_list[pred_char]
        decoded_texts.append(pred_text)
    return decoded_texts

# make predictions
predictions = model.predict(val_gen)

# Decode prediction results
decoded_predictions = decode_predictions(predictions, char_list='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')

start_time = time.time()
correct_predictions = 0
total_images = len(val_image_paths)

for i, path in enumerate(val_image_paths):
    true_label = os.path.basename(path).split('.')[0].replace('-', '')
    predicted_label = decoded_predictions[i]

    print("actual tag:", true_label)
    print("predicted tag:", predicted_label)

    if true_label == predicted_label:
        correct_predictions += 1

accuracy = correct_predictions / total_images
end_time = time.time()
total_time = end_time - start_time

print(f"accuracy: {accuracy * 100:.2f}%")
print(f"total running time: {total_time:.2f}秒")
print(f"the number of images being processed: {total_images}")
