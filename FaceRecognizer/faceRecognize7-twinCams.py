# This Program uses two cameras simultaneously to read the scene
import cv2
print(cv2.__version__)
import numpy as np
import time
dispW=640
dispH=480
flip=2
font=cv2.FONT_HERSHEY_SIMPLEX
dtav=0
#Uncomment These next Two Line for Pi Camera
#camSet='nvarguscamerasrc !  video/x-raw(memory:NVMM), width=3264, height=2464, format=NV12, framerate=21/1 ! nvvidconv flip-method='+str(flip)+' ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'
#cam= cv2.VideoCapture(camSet)

#Or, if you have a WEB cam, uncomment the next line
#(If it does not work, try setting to '1' instead of '0')
cam1=cv2.VideoCapture(0) # Object for the first Webcamera
cam2=cv2.VideoCapture(1) # Object for the second Webcamera
startTime=time.time() # Get the current time
while True:
    ret, frame1 = cam1.read() # Read the frame using the first camera
    frame1=cv2.resize(frame1,(dispW,dispH)) # Resize the image
    ret, frame2 = cam2.read() # Read the frame using the second camera
    frame2=cv2.resize(frame2,(dispW,dispH)) # Resize the image
    frameCombined=np.hstack((frame1,frame2)) # Stack arrays (frame1 & frame2) in sequence horizontally (column wise).
    dt=time.time()-startTime # compute the change in time by subtracting the current time from the previous time
    startTime=time.time() # Reinitialize the start time as current time
    dtav=0.9*dtav+0.1*dt # To control the uprupt variation use a low pass filter
    fps=1/dtav # find the FPS = 1/T
    cv2.rectangle(frameCombined,(0,0),(130,40),(0,0,255),-1) # Draw the solid rectangle
    cv2.putText(frameCombined,str(round(fps,1))+'fps',(0,25),font,0.75,(0,255,255),2) # put the text
    cv2.imshow('WebCam1',frameCombined) # Show the combine from from both cameras
    #cv2.imshow('WebCam1',frame1)
    #cv2.imshow('WebCam2',frame2)
    cv2.moveWindow('WebCam1',0,0)
    #cv2.moveWindow('WebCam2',700,0)
    if cv2.waitKey(1)==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()