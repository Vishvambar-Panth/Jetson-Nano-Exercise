# This is the program to speak the identified objects in the image using google Text-to-speak API
import jetson.inference
import jetson.utils
import numpy as np 
import time
import os
from gtts import gTTS 
import threading

speak=True
item='Welcome Rajapriyan'
confidence=0
itemOld=''

import cv2
print(cv2.__version__)
dispW=1280 #640
dispH=720 #480
flip=2
#Uncomment These next Two Line for Pi Camera
#camSet='nvarguscamerasrc !  video/x-raw(memory:NVMM), width=3264, height=2464, format=NV12, framerate=21/1 ! nvvidconv flip-method='+str(flip)+' ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'
#cam= cv2.VideoCapture(camSet)

def sayItem(): # a function to speak the identified object
    global speak # make speak & item as Global variable
    global item
    while True: 
        if speak==True: # If speak=True, utter the name (item) of the object in the image
            output=gTTS(text=item,lang='en',slow=False) 
            output.save('output.mp3')
            os.system('mpg123 output.mp3')
            speak=False # after utter the item, make speak=False
x=threading.Thread(target=sayItem,daemon=True)
x.start() # start threading

#Or, if you have a WEB cam, uncomment the next line
#(If it does not work, try setting to '1' instead of '0')
cam=cv2.VideoCapture('/dev/video0')
cam.set(cv2.CAP_PROP_FRAME_WIDTH, dispW)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, dispH)
net=jetson.inference.imageNet('googlenet')
font=cv2.FONT_HERSHEY_SIMPLEX
timeMark=time.time()
fpsFilter=0
while True:
    ret, frame = cam.read()
    #frame = cv2.resize(frame, (dispW, dispH), fx = 0, fy = 0, interpolation = cv2.INTER_CUBIC) 
    img=cv2.cvtColor(frame,cv2.COLOR_BGR2RGBA).astype(np.float32)
    img=jetson.utils.cudaFromNumpy(img)
    if speak==False:
        classID,confidence=net.Classify(img,dispW,dispH)
        if confidence>=0.5:
            item=net.GetClassDesc(classID)
            if item!=itemOld:
                speak=True
        if confidence<0.5:
            item=''
        itemOld=item
    dt=time.time()-timeMark
    timeMark=time.time()
    fps=1/dt
    fpsFilter=0.95*fpsFilter+0.05*fps
    cv2.putText(frame,str(round(fpsFilter,1))+' fps '+ ' '+item+' '+str(round(confidence,2)),(0,30),font, 1, (0,0,255),2)
    cv2.imshow('nanoCam',frame)
    cv2.moveWindow('nanoCam',0,0)
    if cv2.waitKey(1)==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()