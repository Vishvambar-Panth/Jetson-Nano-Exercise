import tkinter as tk
from tkinter import Message ,Text
import cv2,os
import shutil
import csv
import numpy as np
from PIL import Image, ImageTk
import pandas as pd
import datetime
import time
import tkinter.ttk as ttk
import tkinter.font as font

window = tk.Tk()
#helv36 = tk.Font(family='Helvetica', size=36, weight='bold')
window.title("Faculty Attendance")

Title = tk.Label(window, text="Face Recognition Based Attendance Management System",fg="black", width=50, height=3, font=('times', 30, 'italic bold underline')) 
Title.place(x=200, y=20)

lbl = tk.Label(window, text="Enter ID",width=20, height=2,fg="red",font=('times', 15, ' bold ') ) 
lbl.place(x=400, y=200)

txt = tk.Entry(window,width=20, fg="red", font=('times', 15, 'bold'))
txt.place(x=700, y=215)

lbl2 = tk.Label(window, text="Enter Name", width=20,fg="red", height=2,font=('times', 15, 'bold')) 
lbl2.place(x=400, y=300)

txt2 = tk.Entry(window,width=20, fg="red",font=('times', 15, 'bold'))
txt2.place(x=700, y=315)

lbl3 = tk.Label(window, text="Notification : ",width=20  ,fg="red", height=2, font=('times', 15, 'bold underline')) 
lbl3.place(x=400, y=400)

message = tk.Label(window, text=""  ,fg="red"  ,width=30  ,height=2, activebackground = "yellow" ,font=('times', 15, ' bold ')) 
message.place(x=700, y=400)

lbl3 = tk.Label(window, text="Attendance : ",width=20  ,fg="red"   ,height=2 ,font=('times', 15, ' bold  underline')) 
lbl3.place(x=400, y=650)

message2 = tk.Label(window, text="" ,fg="red"  ,activeforeground = "green",width=30  ,height=2  ,font=('times', 15, ' bold ')) 
message2.place(x=700, y=650)

def clear():
    txt.delete(0, 'end')    
    res = ""
    message.configure(text= res)

def clear2():
    txt2.delete(0, 'end')    
    res = ""
    message.configure(text= res) 

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
 
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
 
    return False

def TakeImages():        
    Id=(txt.get())
    name=(txt2.get())
    if(is_number(Id) and name.isalpha()):
        cam = cv2.VideoCapture(0)
        #harcascadePath = "haarcascade_frontalface_default.xml"
        #detector=cv2.CascadeClassifier(harcascadePath)
        detector=cv2.CascadeClassifier('/home/rajavel/Desktop/PyProg/Student_Attendance/project_done/haarcascade_frontalface_default.xml')
        sampleNum=0
        while(True):
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in faces:
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)        
                #incrementing sample number 
                sampleNum=sampleNum+1
                #saving the captured face in the dataset folder TrainingImage
                cv2.imwrite("/home/rajavel/Desktop/PyProg/Student_Attendance/project_done/TrainingImage/ "+name +"."+Id +'.'+ str(sampleNum) + ".jpg", gray[y:y+h,x:x+w])
                #display the frame
                cv2.imshow('frame',img)
            #wait for 100 miliseconds 
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            # break if the sample number is morethan 100
            elif sampleNum>60:
                break
        cam.release()
        cv2.destroyAllWindows() 
        res = "Images Saved for ID : " + Id +" Name : "+ name
        row = [Id , name]
        #with open('StudentDetails/StudentDetails.csv','a+') as csvFile:
        with open('/home/rajavel/Desktop/PyProg/Student_Attendance/project_done/StudentDetails/StudentDetails.csv','a+') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
        csvFile.close()
        message.configure(text= res)
    else:
        if(is_number(Id)):
            res = "Enter Alphabetical Name"
            message.configure(text= res)
        if(name.isalpha()):
            res = "Enter Numeric Id"
            message.configure(text= res)

takeImg = tk.Button(window, text="Take Images", command=TakeImages  ,fg="red"   ,width=20  ,height=3, activebackground = "Red" ,font=('times', 15, ' bold '))
takeImg.place(x=200, y=500)
'''
def TrainImages():
    #recognizer =cv2.face.createLBPHFaceRecognizer()
    #recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer=cv2.createLBPHFaceRecognizer()
    detector =cv2.CascadeClassifier('/home/rajavel/Desktop/PyProg/Student_Attendance/project_done/haarcascade_frontalface_default.xml')
    faces,Id = getImagesAndLabels("/home/rajavel/Desktop/PyProg/Student_Attendance/project_done/TrainingImage")
    recognizer.train(faces, np.array(Id))
    recognizer.save("/home/rajavel/Desktop/PyProg/Student_Attendance/project_done/TrainingImageLabel/Trainner.yml")
    res = "Image Trained"#+",".join(str(f) for f in Id)
    message.configure(text= res)

trainImg = tk.Button(window, text="Train Images", command=TrainImages  ,fg="red"    ,width=20  ,height=3, activebackground = "Red" ,font=('times', 15, ' bold '))
trainImg.place(x=500, y=500)
'''
window.mainloop()



