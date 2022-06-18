# -*- coding: utf-8 -*-
"""
Created on Fri May 27 14:39:32 2022

@author: Alan
"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, Flatten, Dense
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization, Dropout
from tensorflow.keras.layers import LeakyReLU,ReLU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential

classes = {'cheetah':0, 'fox':1, 'hyena':2, 
           'lion':3, 'tiger':4, 'wolf':5}
           
# =============================================================================
# x_train = np.load('x_train.npy')
# y_train = np.load('y_train.npy')
# print('Data Loaded')
# x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.2)
# =============================================================================
a= tf.test.is_gpu_available()
b=tf.config.list_physical_devices('GPU')
print(a,b)
#%%
# Example-Flip
def augmentation(images, lables): # flip
    images_ = np.concatenate([images, np.flip(images, axis=1)])
    lables_ = np.concatenate([lables,lables])
    return images_, lables_

x_train, y_train = augmentation(x_train, y_train)
    
#%%    
from tensorflow.keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(
     rotation_range=20,
     width_shift_range=0.2,
     height_shift_range=0.2,
    zoom_range=(0.8,1.4),
     horizontal_flip=True
    )
# =============================================================================
# for img in x_train:
#     datagen.fit(np.expand_dims(img,0))
#     aug_gen = datagen.flow(np.expand_dims(img,0),np.array([1]))
#     plt.figure(figsize = (12,6))
#     for i in range(15):
#         img_,_ = aug_gen.next()
#         x_train = np.append(x_train, img_, axis= 0)
#     break
# =============================================================================
datagen.fit(x_train)
#%%
def build_model(input_shape = (224,224,3)):
    input_layer = Input(shape = input_shape)
    x = input_layer
    kernel_size = 5
    x = Conv2D(6, kernel_size, padding = 'same', activation='relu')(input_layer)
    x = MaxPooling2D((2, 2))(x)
    
    x = Conv2D(16, kernel_size, activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)

    x = Flatten()(x)
    x = Dense(120, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    output_layer = Dense(6, activation = 'softmax')(x)   
    #super().__init__(input_layer, output_layer, name = 'LeNet')    
    model = Model(input_layer, output_layer, name = 'wild_Animal')
    model.compile(optimizer=Adam(lr = 0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model


#%% Build model
model = build_model(input_shape = x_train.shape[1:])
model.summary()

early_stop = EarlyStopping(monitor = 'val_loss', patience=3, min_delta = 0.01)
history = model.fit(datagen.flow(x_train,y_train,
                    batch_size = 16),
                    epochs = 40,
                    #validation_split = 0.2,
                    validation_data = (x_val,y_val)#,
                    #callbacks = [early_stop]
                    )  


#%% Model evaluation
plt.plot(history.history['loss'], label = 'loss')
plt.plot(history.history['val_loss'], label = 'val_loss')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label = 'acc')
plt.plot(history.history['val_accuracy'], label = 'val_acc')
plt.legend()
plt.show()
#%%
model.save('model.h5')
# In[15]:
pred_train = model.predict(x_train).argmax(1)
pred_test = model.predict(x_val).argmax(1)
#is the x-test xval?
print('Train Accuracy:', accuracy_score(pred_train, y_train))
print('Test Accuracy:', accuracy_score(pred_test, y_val))
