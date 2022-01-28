# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 12:46:38 2021

@author: rashmi
"""


import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
train_data = pd.read_csv('C:/Users/rashmi/Documents/Python_Scripts/TrainIJCNN2013/TrainIJCNN2013/recog/Train1.csv')
test_data = pd.read_csv('C:/Users/rashmi/Documents/Python_Scripts/TrainIJCNN2013/TrainIJCNN2013/recog/Test.csv')

# Load the labels
train_labels = train_data['ClassId']
test_labels = test_data['ClassId']

# Load the images
train_image_paths = train_data['Path'].apply(lambda x: 'C:/Users/rashmi/Documents/Python_Scripts/TrainIJCNN2013/TrainIJCNN2013/recog/' + x).tolist()
test_image_paths = test_data['Path'].apply(lambda x: 'C:/Users/rashmi/Documents/Python_Scripts/TrainIJCNN2013/TrainIJCNN2013/recog/' + x).tolist()
train_images = np.array([cv2.cvtColor(cv2.resize(cv2.imread(fname1), (32, 32), interpolation = cv2.INTER_AREA), cv2.COLOR_BGR2RGB) for fname1 in train_image_paths])
test_images = np.array([cv2.cvtColor(cv2.resize(cv2.imread(fname), (32, 32), interpolation = cv2.INTER_AREA), cv2.COLOR_BGR2RGB) for fname in test_image_paths])
train_images = np.sum(train_images/3, axis=3, keepdims=True)
train_images = train_images / 255.0

test_images = np.sum(test_images/3, axis=3, keepdims=True)
test_images = test_images / 255.0
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters = 6, kernel_size = (5, 5), strides=(1, 1), padding='valid', 
                        activation='relu', data_format = 'channels_last', input_shape = (32, 32, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(16, (5, 5), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(43, activation='softmax')
])
#model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy']) 
#model.compile(optimizer='adam', loss=tf.keras.losses.sparse_categorical_crossentropy(from_logits=True), metrics=['accuracy'])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# Train
#conv=model.fit(train_images, train_labels,epochs= 30, steps_per_epoch=10, validation_data= (test_images, test_labels), validation_steps=2)
conv = model.fit(train_images, train_labels, batch_size= 128, epochs=30, validation_data=(test_images, test_labels))