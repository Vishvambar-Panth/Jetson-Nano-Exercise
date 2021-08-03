# This program creates a thread using Python Thread module to acquire a image from two cameras simultaneously without a delay
from threading import Thread # threading module in Python, emulating a subset of Java's threading model
import cv2
import time
import numpy as np


class vStream: # Define a class named vStream
    def __init__(self,src,width,height): # Initialization function. self - to hold object (cam0 & cam1). src - camera source. [width, height] - represents camera size
        self.width=width # Object (cam0/1) width
        self.height=height # Object (cam0/1) height
        self.capture=cv2.VideoCapture(src) # object (cam0/1) videoCapture. src - camera source 
        self.thread=Thread(target=self.update,args=()) # A class that represents a thread of control.
        self.thread.daemon=True
        self.thread.start() # Start the thread's activity. It must be called at most once per thread object. This method will raise a RuntimeError if called more than once on the same thread object.
    def update(self): # a function in the class
        while True:
            _,self.frame=self.capture.read() # read the frame for the given object (cam0/1) ()= cam.read())
            self.frame2=cv2.resize(self.frame,(self.width,self.height)) # Resize the acquired frame to the specified width and height
    def getFrame(self): # a function to return the acquired frame
        return self.frame2  # frame - original frame. frame-2 - resized frame

dispW=640 # Frame width
dispH=480 # Frame height
cam0=vStream(0,dispW,dispH) # Get the frame using the object cam0 with specified width and height
cam1=vStream(1,dispW,dispH) # Get the frame using the object cam1 with specified width and height
font=cv2.FONT_HERSHEY_SIMPLEX # define the font
startTime=time.time() # StartTime = current time
dtav=0 
while True:
    try:
        myFrame0=cam0.getFrame() # Get frame from the first camera
        myFrame1=cam1.getFrame() # Get frame from the second camera
        myFrame3=np.hstack((myFrame0,myFrame1)) # Stack the frames
        #cv2.imshow('WebCam0',myFrame0)
        #cv2.imshow('WebCam1',myFrame1)
        dt=time.time()-startTime # find the difference in time
        startTime=time.time() # Revise the start time for the next cycle
        dtav=0.9*dtav+0.1*dt # To control the uprupt variation use a LPF
        fps=1/dtav
        cv2.rectangle(myFrame3,(0,0),(100,40),(0,0,255),-1) # Draw a solid rectangle
        cv2.putText(myFrame3,str(round(fps,1))+'fps',(0,25),font,0.75,(0,255,255),2) #put the frame rate as text
        cv2.imshow('Combo',myFrame3)
        cv2.moveWindow('Combo',0,0)

    except:
        print('frame not available') # if you couldn't read frame print "frame not available"
    if cv2.waitKey(1)==ord('q'):
        cam0.capture.release() # Release the cameras and destroy all the windows
        cam1.capture.release()
        cv2.destroyAllWindows()
        exit(1)
        break
        

