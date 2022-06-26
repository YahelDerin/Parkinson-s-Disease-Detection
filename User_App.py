# -*- coding: utf-8 -*-
"""
This file is responsible for communicating with the user
"""
import time

def check_option(flag_option1):
    # This fanction check the option that the user choose
    option = text()  #user's selction      
    while option!="1" and flag_option1==False:
        print("In the first time you must to choose in option number 1. Please try again\n")
        time.sleep(2)
        option = text()
    if option=="1" and flag_option1==False:
        flag_option1 = True
        return "1",flag_option1
    while option=="1" and flag_option1==True:
        print("Error: you can't choose this option more than once")
        print("try to choose again")
        option = text()
    return option, flag_option1

                 
def text():
    # This function shows the program options, and ask the user to choose one
    print(" Here are the options of the program:")
    time.sleep(3)
    print("  1) Prepare the Data")
    time.sleep(2)
    print("  2) Train the Model")
    time.sleep(2)
    print("  3) Test the Model")
    time.sleep(2)
    print("  4) Predict an Image\n\n")
    time.sleep(3)
    option = input("Please enter the number of the option you choose:\n  (note:  If you are running the project for the first time, you must select the first option)\n")
    if option!="1" and option!="2" and option!="3" and option!="4":
        print("This option doesn't exist. Please try to choose again")
        text()
    return option


