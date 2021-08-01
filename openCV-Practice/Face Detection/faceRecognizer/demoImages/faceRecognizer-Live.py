import pickle
import face_recognition
import os
import cv2
print(cv2.__version__)

Encodings = []
Names = []
j=0

with open('train.pkl','rb') as f:
    Names=pickle.load(f)
    Encodings = pickle.load(f)

dispW=640
dispH=480
flip=2
camset='nvarguscamerasrc sensor_id=0 ! video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=21/1 ! nvvidconv flip-method='+str(flip)+' ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'
cam=cv2.VideoCapture(camset)
font = cv2.FONT_HERSHEY_SIMPLEX
while True:
    ret,frame=cam.read()
    frameSmall = cv2.resize(frame,(0,0),fx=0.33,fy=0.33)
    frameRGB = cv2.cvtColor(frameSmall,cv2.COLOR_BGR2RGB)
    facePositions = face_recognition.face_locations(frameRGB)   
    allEncodings = face_recognition.face_encodings(frameRGB,facePositions)

    for (top,right,bottom,left),faceEncoding in zip(facePositions,allEncodings):
        name='Unknown Person'
        matches = face_recognition.compare_faces(Encodings,faceEncoding)
        if True in matches:
            match_index = matches.index(True)
            name = Names[match_index]
        top=top*3
        bottom=bottom*3
        left=left*3
        right=right*3
        cv2.rectangle(frame,(left,top),(right,bottom),(0,0,255),2)
        cv2.putText(frame,name,(left,top-6),font,1,(0,255,255),2)

    
    cv2.imshow('Window',frame)
    cv2.moveWindow('Window',0,0)

   
    if cv2.waitKey(0)==ord('q'):
        break
cv2.destroyAllWindows()
cam.release()