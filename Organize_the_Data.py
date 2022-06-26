# -*- coding: utf-8 -*-
"""
This file download and prepare the dataset for the model training.
"""

import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import os
from zipfile import ZipFile 
  

def extract_zip_file():
    # this function extract the zip file that contains the dataset
    zip_path = input("Please Enter the path of the Zip File") #the zip file path
    with ZipFile(zip_path, 'r') as zip: 
        # extract all files to another directory
        zip.extractall('new_dataset')
    print("\n\nFinish to extract the files\n\n")


def loading_dataset(path_to_files):
    #this function loading the dataset, and prepare it to train the model 
    array_of_images = [] #a numpy array containing the single_array arrays
    for file in os.listdir(path_to_files):
        #if "direction.png" in file: # to check if file has a certain name   
            single_im = Image.open(path_to_files +'/'+file) #PNG image
            single_im = single_im.resize((128,128))
            single_array = np.array(single_im) #single_image as a numpy array
            array_of_images.append(single_array)            
    array_of_images = np.array(array_of_images)
    np.savez("all_images.npz",array_of_images) # save all in one file
    
    data_train = np.load('all_images.npz', allow_pickle=True)
    x_list = data_train['arr_0'] #the loaded images list
    y_list = [] #the labels list 
    for file in os.listdir(path_to_files):
        if 'HE' in file:
            y_list.append(0)
        else:
            y_list.append(1)
     
    np.array(to_categorical(y_list, dtype='uint8'))
    print("\n\nFinish to load all the images\n\n")
    return x_list, y_list



def augmenting_the_dataset(x_list, y_list, num):
    #The augmentation function
    data_generator = ImageDataGenerator(rotation_range=360, 
                                        width_shift_range=0.0, 
                                        height_shift_range=0.0, 
                                         brightness_range=[0.5, 1.5],
                                        horizontal_flip=True, 
                                        vertical_flip=True)
    
    x = list(x_list)
    y = list(y_list)
    
    x_aug = [] #the new images list
    y_aug = [] #the list of the labels of the new images
    
    for (i, v) in enumerate(y):
        x_img = x[i]
        x_img = np.array(x_img)
        x_img = np.expand_dims(x_img, axis=0)
        aug_iter = data_generator.flow(x_img, batch_size=1, shuffle=True)
        for j in range(num):
            aug_image = next(aug_iter)[0].astype('uint8')
            x_aug.append(aug_image)
            y_aug.append(v)
    print("num of new images: ", len(x_aug))
    
    x_list = x + x_aug
    y_list = y + y_aug
    print("the num of the total images: ", len(x_list))
    print("\n\nAugmentation is finished\n\n")
    return x_list, y_list



def preprocessing_the_images(x_train, y_train, x_test, y_test):
    #this function convert the training set and the testing set to numpy arrays
    train =[] #list of the training images in gray 
    for i in range(len(x_train)):
        img = x_train[i]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        train.append(img)
    x_train=train[:]
    
    test =[] #list of the testing images in gray
    for i in range(len(x_test)):
        img = x_test[i]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        test.append(img)
    x_test = test[:]
    
    
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    
    x_train = x_train/255.0
    x_test = x_test/255.0
    
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    
    label_encoder = LabelEncoder()
    y_test = label_encoder.fit_transform(y_test)
    print("\n\nThe dataset is prepare for training the model\n\n")
    return x_train, y_train, x_test, y_test