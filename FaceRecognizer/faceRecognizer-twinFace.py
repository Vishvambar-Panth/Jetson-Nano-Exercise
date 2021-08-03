# This program Recognize the faces using two cameras simultaneously. Refer a program "faceRecognizer-11twinCams.py" for more info on using two cameras with treading concept.
from threading import Thread #A class that represents a thread of control.
import cv2
import time
import numpy as np
import face_recognition 
import pickle


with open('train.pkl','rb') as f: # Trained model
    Names=pickle.load(f)
    Encodings=pickle.load(f)


class vStream:
    def __init__(self,src,width,height):
        self.width=width
        self.height=height
        self.capture=cv2.VideoCapture(src)
        self.thread=Thread(target=self.update,args=())
        self.thread.daemon=True
        self.thread.start()
    def update(self):
        while True:
            _,self.frame=self.capture.read()
            self.frame2=cv2.resize(self.frame,(self.width,self.height))
    def getFrame(self):
        return self.frame2

dispW=640
dispH=480
cam0=vStream(0,dispW,dispH)
cam1=vStream(1,dispW,dispH)
dtav=0
font=cv2.FONT_HERSHEY_SIMPLEX
startTime=time.time()
scaleFactor=0.33
while True:
    try:
        myFrame0=cam0.getFrame() # Get the frame from camera-1
        myFrame1=cam1.getFrame() # Get the frame from camera-2
        myFrame3=np.hstack((myFrame0,myFrame1)) # Stack the frames
        frameRGB=cv2.cvtColor(myFrame3,cv2.COLOR_RGB2BGR) #convert the combined frame from RGB to BGR
        frameRGBsmall=cv2.resize(frameRGB,(0,0),fx=scaleFactor,fy=scaleFactor) # Resize the frame using the scale factor
        facePositions=face_recognition.face_locations(frameRGBsmall,model='cnn') # Find the face_locations using cnn model
        allEncodings=face_recognition.face_encodings(frameRGBsmall,facePositions) # Encode the faces in Test Image
        for (top,right,bottom,left),face_encoding in zip(facePositions,allEncodings):
            name='Unknown Person'
            matches=face_recognition.compare_faces(Encodings,face_encoding) # Compare the face encodings in Training and Tesing Images
            if True in matches:
                first_match_index=matches.index(True)
                name=Names[first_match_index]
                print(name)
            top=int(top/scaleFactor) # Resize the image back to original size
            left=int(left/scaleFactor)
            right=int(right/scaleFactor)
            bottom=int(bottom/scaleFactor)
            cv2.rectangle(myFrame3,(left,top),(right,bottom),(0,0,255),2)
            cv2.putText(myFrame3,name,(left,top-6),font,0.75,(0,255,255),2)
        

        #cv2.imshow('WebCam0',myFrame0)
        #cv2.imshow('WebCam1',myFrame1)
        dt=time.time()-startTime
        startTime=time.time()
        dtav=0.9*dtav+0.1*dt
        fps=1/dtav
        cv2.rectangle(myFrame3,(0,0),(100,40),(0,0,255),-1)
        cv2.putText(myFrame3,str(round(fps,1))+'fps',(0,25),font,0.75,(0,255,255),2)
        cv2.imshow('Combo',myFrame3)
        cv2.moveWindow('Combo',0,0)

    except:
        print('frame not available')
    if cv2.waitKey(1)==ord('q'):
        cam0.capture.release()
        cam1.capture.release()
        cv2.destroyAllWindows()
        exit(1)
        break
        

