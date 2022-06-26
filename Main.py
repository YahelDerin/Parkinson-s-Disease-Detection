# -*- coding: utf-8 -*-
"""
This file manages the entire program. from here commands are sent to the other files
"""
import Organize_the_Data
import Show_Data
import the_Model
import User_App
import Test_the_Model


import os
import time
 

def main():
    print("Hi! welcome to Parkinson's Disease Detection")
    time.sleep(3)
    flag_option1=False #False-if user didn't choose number 1, True-if user choose number 1
    user_want ='' #if user want to exit the program, it will be equal to 'Exit'
    while user_want!='Exit':
        option, flag_option1 = User_App.check_option(flag_option1) #option=user selection
        if option=="1":
            Organize_the_Data.extract_zip_file()
            dir_path = input("Please enter the new dataset path") #the dataset path
            x_train, y_train = Organize_the_Data.loading_dataset(os.path.join(dir_path +'/training')) #training set
            x_test, y_test = Organize_the_Data.loading_dataset(os.path.join(dir_path +'/testing'))    #testing set
            Show_Data.data_distribution(y_train, y_test,'before')
            x_train, y_train = Organize_the_Data.augmenting_the_dataset(x_train, y_train,70)
            x_test, y_test =  Organize_the_Data.augmenting_the_dataset(x_test, y_test,20)
            x_train, y_train, x_test, y_test = Organize_the_Data.preprocessing_the_images(x_train, y_train, x_test, y_test)
            Show_Data.data_distribution(y_train, y_test, 'after')
            print("The dataset is prepare")
        if option=="2":
            model= the_Model.parkinson_disease_detection_model(input_shape=(128, 128, 1)) #contains the model
            model.summary()
            hist = model.fit(x_train, y_train, batch_size=32, epochs=30, validation_data=(x_test, y_test)) #the trained model
            Show_Data.loss_and_accuracy_plot(hist)
            model.save('parkinson_disease_detection.h5') #saves the model
            print("The model done to fit and saved")
        if option=="3":
            model = Test_the_Model.upload_the_model()
            Test_the_Model.test_model(model, x_test, y_test) 
            print("The model test completed")
        if option=="4": 
            Organize_the_Data.extract_zip_file()
            model = Test_the_Model.upload_the_model()
            Test_the_Model.Testing_Model_on_Images(model)
            print("The image detection completed")
        time.sleep(3)
        print("If you want to exit the program - please write 'Exit'")
        print("Else press ENTER")
        user_want = input()

if __name__ == "__main__":
    main()