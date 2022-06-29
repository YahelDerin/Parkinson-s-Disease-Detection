# -*- coding: utf-8 -*-
"""
This file is responsible for creating the model
"""
from tensorflow.keras.layers import Input, Conv2D, Dropout, Flatten, Dense, MaxPool2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import tensorflow as tf


def parkinson_disease_detection_model(input_shape=(128, 128, 1)):
    #The model
    regularizer = tf.keras.regularizers.l2(0.001)
    model = Sequential()
    model.add(Input(shape=input_shape))
    
    model.add(Conv2D(128, (3, 3), padding='same', strides=(1, 1), name='conv1', activation='relu', 
                     kernel_initializer='glorot_uniform', kernel_regularizer=regularizer))
    model.add(MaxPool2D((5, 5), strides=(2, 2)))
    
    """    model.add(Conv2D(128, (3, 3), padding='same', strides=(1, 1), name='conv1', activation='relu', 
                         kernel_initializer='glorot_uniform', kernel_regularizer=regularizer))
        model.add(MaxPool2D((2, 2), strides=(3, 3)))
    """
    
    model.add(Conv2D(64, (3, 3), padding='same', strides=(1, 1), name='conv2', activation='relu', 
                     kernel_initializer='glorot_uniform', kernel_regularizer=regularizer))
    model.add(MaxPool2D((3, 3), strides=(2, 2)))

    model.add(Conv2D(32, (3, 3), padding='same', strides=(1, 1),activation='relu', 
                     kernel_initializer='glorot_uniform', kernel_regularizer=regularizer))
    model.add(MaxPool2D((2, 2), strides=(1, 1)))

    model.add(Flatten())
    model.add(Dropout(0.2))
    #model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu', kernel_initializer='glorot_uniform'))
    #model.add(Dropout(0.2))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu', kernel_initializer='glorot_uniform'))
    #model.add(Dropout(0.2))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation='relu', kernel_initializer='glorot_uniform'))
    model.add(Dropout(0.2))
    #model.add(Dropout(0.2))
    #model.add(Dense(8, activation='relu', kernel_initializer='glorot_uniform'))
    #model.add(Dropout(0.2))
    #model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
    #, kernel_initializer='glorot_uniform', name='fc3'))
    
    optimizer = Adam(3.15e-5)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model
    """
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu', kernel_initializer='glorot_uniform', name='fc1'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu', kernel_initializer='glorot_uniform', name='fc2'))
    model.add(Dense(16, activation='relu', kernel_initializer='glorot_uniform', name='fc3'))
    model.add(Dense(1, activation='sigmoid'))
    
    optimizer = Adam(3.15e-5)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model
    """
    

    