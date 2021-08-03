'''
This project is an extention of the previous deepLearning-1a.py project
here openCV is used to acquire as well as to display the image 
but processed using Jetson inference and utilities
-------------------------------------------------------------------------------------------
'''
import jetson.inference
import jetson.utils
import cv2
import numpy as np
import time
width=640
height=480
dispW=width
dispH=height
flip=2
#camSet='nvarguscamerasrc !  video/x-raw(memory:NVMM), width=3264, height=2464, format=NV12, framerate=21/1 ! nvvidconv flip-method='+str(flip)+' ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'
#cam1=cv2.VideoCapture(camSet)
cam1=cv2.VideoCapture('/dev/video0')
cam1.set(cv2.CAP_PROP_FRAME_WIDTH,dispW)
cam1.set(cv2.CAP_PROP_FRAME_HEIGHT,dispH)
#cam= cv2.VideoCapture(camSet)
#cam=jetson.utils.gstCamera(width,height,'/dev/video0')
#cam=jetson.utils.gstCamera(width,height,'0') # PiCam
#display=jetson.utils.glDisplay()
font=jetson.utils.cudaFont()
net=jetson.inference.imageNet('googlenet')
font=cv2.FONT_HERSHEY_SIMPLEX
timeMark=time.time()
fpsFilter=0
while True:
    #frame,width,height=cam.CaptureRGBA(zeroCopy=1)
    _,frame=cam1.read()
    img=cv2.cvtColor(frame,cv2.COLOR_BGR2BGRA).astype(np.float32)# Convert BGR to BGRA
    img=jetson.utils.cudaFromNumpy(img) # Transfer image from numpy to cuda
    classID,confidence=net.Classify(img,width,height) # Classify the acquired image against the trained model
    item=net.GetClassDesc(classID) # Get the Class description as item
    dt=time.time()-timeMark # Compute the change in time
    fps=1/dt # compute the fps
    fpsFilter=0.95*fpsFilter+0.05*fps # Apply the LPF
    timeMark=time.time() # Restart the time
    #frame=jetson.utils.cudaToNumpy(frame,width,height,4)
    #frame=cv2.cvtColor(frame,cv2.COLOR_RGBA2BGR).astype(np.uint8)
    cv2.putText(frame,str(round(fpsFilter,1))+' fps '+item,(0,30),font,1,(0,0,255),2)
    cv2.imshow('RecoCam',frame)
    cv2.moveWindow('RecoCam',0,0)
    if cv2.waitKey(1)==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
    
