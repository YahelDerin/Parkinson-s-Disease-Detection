# -*- coding: utf-8 -*-
"""
This file is responsible for uploading the saved model,
examining the model by the test set and is responsible for predicting the image selected by the user.
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
import keras


def upload_the_model():
    #This function upload the saved model from the disk
    path = input("please enter the path's file of the save model") #path of the save model
    model = keras.models.load_model(path) #the saved model
    return model

    
def test_model(model, x_test, y_test):
    #This function test the model with the testing set
    results = model.evaluate(x_test, y_test, batch_size=32) #training results.
    print("test loss: {}, test acc: {}".format(results[0], results[1]))
    
    
def testing_model_on_images(model):
    #this function is responsible to ask the user for an image, and predict it
    labels = ['Healthy', 'Parkinson'] #the 2 images labels
    path = input("Please enter the image path you want to predict") #the path of the single image
    image = cv2.imread(path)
    
    image = cv2.resize(image, (128, 128))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=-1)
    
    ypred = model.predict(image) #the predicted results.
    
    plt.figure(figsize=(2, 2))
    img = np.squeeze(image, axis=0)
    plt.imshow(img)
    plt.axis('off')
    plt.title(f'Prediction by the model: {labels[np.argmax(ypred[0], axis=0)]}')
    plt.show()