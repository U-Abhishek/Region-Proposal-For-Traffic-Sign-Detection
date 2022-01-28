# -*- coding: utf-8 -*-
"""
Created on Mon May 18 19:36:16 2020

@author: Ruchir
"""
import pickle
with open('train02Dec.pickle', 'rb') as f:
     [train_images,train_labels] = pickle.load(f)
#load('train_images')
#load(train_labels)
from tensorflow.keras.layers import Dense
#from keras import Model
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.vgg16 import VGG16
#vggmodel = VGG16(weights='imagenet', include_top=True)
import tensorflow as tf
from tensorflow.keras.models import Model
# for layers in (vggmodel.layers)[:15]:
#     print(layers)
#     layers.trainable = False
# X= vggmodel.layers[-2].output
# predictions = Dense(2, activation="softmax")(X)
# inputs=vggmodel.input
# outputs=predictions
# model_final = tf.keras.Model(inputs=inputs,outputs=outputs)
# opt = Adam(lr=0.0001)
# model_final.compile(loss = tf.keras.losses.categorical_crossentropy, optimizer = opt, metrics=["accuracy"])
#model_final.summary()
from tensorflow.keras.models import load_model
model_final = load_model('try_new_model.h5')
#model_final.compile(loss = tf.keras.losses.categorical_crossentropy, optimizer = opt, metrics=["accuracy"])
model_final.summary()
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#import train_images
#import train_labels
X_new = np.array(train_images)
y_new = np.array(train_labels)


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
class MyLabelBinarizer(LabelBinarizer):
    def transform(self, y):
        Y = super().transform(y)
        if self.y_type_ == 'binary':
            return np.hstack((Y, 1-Y))
        else:
            return Y
    def inverse_transform(self, Y, threshold=None):
        if self.y_type_ == 'binary':
            return super().inverse_transform(Y[:, 0], threshold)
        else:
            return super().inverse_transform(Y, threshold)
lenc = MyLabelBinarizer()
Y =  lenc.fit_transform(y_new)
X_train, X_test , y_train, y_test = train_test_split(X_new,Y,test_size=0.10)
trdata = ImageDataGenerator(horizontal_flip=False, vertical_flip=False, rotation_range=90)
traindata = trdata.flow(x=X_train, y=y_train)
tsdata = ImageDataGenerator(horizontal_flip=False, vertical_flip=False, rotation_range=90)
testdata = tsdata.flow(x=X_test, y=y_test)
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
checkpoint = ModelCheckpoint("try_new_model.h5", monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_loss', min_delta=0, patience=100, verbose=1, mode='auto')
model_final.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) 
#hist = model_final.fit_generator(generator= traindata, steps_per_epoch= 1, epochs= 2, validation_data= testdata, validation_steps=2, callbacks=[checkpoint,early])
hist = model_final.fit_generator(generator= traindata, steps_per_epoch= 10 , epochs= 30, validation_data= testdata, validation_steps=2,callbacks=[checkpoint,early])