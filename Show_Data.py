# -*- coding: utf-8 -*-
"""
This file contains all the functions that print visual information about the dataset and the program.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time


def data_distribution(y_train, y_test, situation):
    #The function shows a bar graph showing the number of patients against to the number of healthy people in the same set.
    unique_train, count_train = np.unique(y_train, return_counts=True)
    plt.figure(figsize=(10, 5))
    title = "Number of training images per category " + situation + " augmentation:"
    sns.barplot(unique_train, count_train).set_title(title)
    plt.show()
    time.sleep(7)

    unique_test, count_test = np.unique(y_test, return_counts=True)
    plt.figure(figsize=(10, 5))
    title = "Number of testing images per category " + situation + " augmentation:"
    sns.barplot(unique_test, count_test).set_title(title)
    plt.show()
    time.sleep(7)
        
    
def loss_and_accuracy_plot(hist):
    #This function shows the model's accuracy and loss graphs.
    plt.figure(figsize=(10, 10))
    plt.plot(hist.history['accuracy'], label='Train_accuracy')
    plt.plot(hist.history['val_accuracy'], label='Test_accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc="upper left")
    plt.show()
    
    plt.figure(figsize=(10, 10))
    plt.plot(hist.history['loss'], label='Train_loss')
    plt.plot(hist.history['val_loss'], label='Test_loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc="upper left")
    plt.show()