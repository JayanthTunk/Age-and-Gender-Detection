from tkinter import *
import tkinter as tk
from PIL import Image,ImageTk
import cv2
import argparse
import time
import math
import numpy as np
import matplotlib.pyplot as plt

splash_root = Tk()
splash_root.title("Color Recognition")
splash_root.geometry("1280x720")

upload= Image.open("Wait.jpeg")
image=ImageTk.PhotoImage(upload)
label= Label(splash_root,image=image,height = 0, width =0)
label.place(x=0,y=0)


#splash_label = Label(splash_root,text="Wait if you are not a robot",font=30)
#splash_label.pack()


def mainWin():
   splash_root.destroy()
  # win= Tk()
   #win.title("Main Window")
  # win.geometry("700x200")
  # win_label= Label(win, text= "Close to st", font= ('Helvetica', 25), fg= "red").pack(pady=20)

#Splash Window Timer

splash_root.after(7000, mainWin)

mainloop()

def FaceBox(net, frame,conf_threshold = 0.75):
    FD = frame.copy()
    FH = FD.shape[0]
    FW = FD.shape[1]
    blob = cv2.dnn.blobFromImage(FD,1.0,(300,300),
                                 [104, 117, 123], True, False)

    net.setInput(blob)
    Det = net.forward()
    bboxes = []

    for i in range(Det.shape[2]):
        confi = Det[0,0,i,2]
        if confi > conf_threshold:
            x1 = int(Det[0,0,i,3]* FW)
            y1 = int(Det[0,0,i,4]* FH)
            x2 = int(Det[0,0,i,5]* FW)
            y2 = int(Det[0,0,i,6]* FH)
            bboxes.append([x1,y1,x2,y2])
            cv2.rectangle(FD,(x1,y1),(x2,y2),(255,255,255),
                          int(round(FH/150)),8)

    return FD , bboxes


GP = "gender.prototxt"
GM = "gender.caffemodel"

FP = "face_detector.pbtxt"
FM = "face_detector.pb"

AP = "age.prototxt"
AM = "age.caffemodel"

AN = cv2.dnn.readNet(AM,AP)
GN = cv2.dnn.readNet(GM, GP)
FN = cv2.dnn.readNet(FM, FP)

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
List_Ages = ['(0-3)', '(3-7)', '(7-14)', '(14-22)', '(22-35)', '(35-44)', '(44-57)', '(57-100)']
List_Genders = ['Male', 'Female']



cap = cv2.VideoCapture(0)
padding = 20

while cv2.waitKey(1) < 0:
    
    t = time.time()
    hasFrame , frame = cap.read()

    if not hasFrame:
        cv2.waitKey()
        break
    
    small_frame = cv2.resize(frame,(0,0),fx = 0.5,fy = 0.5)

    frameFace ,bboxes = FaceBox(FN,small_frame)
    if not bboxes:
        print("No face Detected, Checking next frame")
        continue
    for bbox in bboxes:
        face = small_frame[max(0,bbox[1]-padding):min(bbox[3]+padding,frame.shape[0]-1),
                max(0,bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        GN.setInput(blob)
        genderPreds = GN.forward()
        gender = List_Genders[genderPreds[0].argmax()]
        print("Gender : {}, conf = {:.3f}".format(gender, genderPreds[0].max()))

        AN.setInput(blob)
        APred = AN.forward()
        age = List_Ages[APred[0].argmax()]
        print("Age Output : {}".format(APred))
        print("Age : {}, conf = {:.3f}".format(age, APred[0].max()))

        label = "{},{}".format(gender, age)
        cv2.putText(frameFace, label, (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv2.LINE_AA)
        cv2.imshow("Age Prediction - Jayanth Tunk", frameFace)
       
    print("time : {:.3f}".format(time.time() - t))


    
