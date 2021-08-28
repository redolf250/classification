import logs.logs as log
import numpy as np
import matplotlib as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.activations import relu, softmax
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator

log.scanning()
log.importing()

log.resolving_path()
train_path = 'C:/Users/Maven_central/Desktop/Tensorflow/classifier/data/train'
valid_path = 'C:/Users/Maven_central/Desktop/Tensorflow/classifier/data/validation'
log.resolving_path_done()

# log.image_preprocessing()
# train = ImageDataGenerator().flow_from_directory(directory=train_path, target_size=(200, 200), classes=['cat', 'dog'])
# valid = ImageDataGenerator().flow_from_directory(directory=valid_path, target_size=(200, 200), classes=['cat', 'dog'])
# log.image_preprocessing_done()

log.image_preprocessing()
train_set = ImageDataGenerator().flow_from_directory(directory=train_path, target_size=(200, 200), batch_size=10)
valid_set = ImageDataGenerator().flow_from_directory(directory=valid_path, target_size=(200, 200), batch_size=10)
log.image_preprocessing_done()

log.creating_model()
model = Sequential([
    Conv2D(filters=32, kernel_size=(3, 3), input_shape=(200, 200, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2), strides=2),
    Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2), strides=2),
    Flatten(),
    Dense(units=2, activation='softmax'),
])
log.creating_model_done()

model.summary()

log.compile_model()
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
log.compiling_done()

#log.fitting_model()
#model.fit(x=train_set, validation_data=valid_set, epochs=10, verbose=1)
#log.fitting_done()

