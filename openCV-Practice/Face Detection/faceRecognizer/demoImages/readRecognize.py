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

font = cv2.FONT_HERSHEY_SIMPLEX
testImage = face_recognition.load_image_file('/home/jetson/Desktop/openCV-Practice/Face Detection/faceRecognizer/demoImages/unknown/u1.jpg')  
facePositions = face_recognition.face_locations(testImage)      
allEncodings = face_recognition.face_encodings(testImage,facePositions)

testImage = cv2.cvtColor(testImage,cv2.COLOR_RGB2BGR)

for (top,right,bottom,left),faceEncoding in zip(facePositions,allEncodings):
    name='Unknown Person'
    matches = face_recognition.compare_faces(Encodings,faceEncoding)
    if True in matches:
        match_index = matches.index(True)
        name = Names[match_index]
    cv2.rectangle(testImage,(left,top),(right,bottom),(0,0,255),2)
    cv2.putText(testImage,name,(left,top-6),font,1,(0,255,255),2)


frame=cv2.resize(testImage, (640,480))
cv2.imshow('Window',frame)
cv2.moveWindow('Window',0,0)

print('Showing Window')
if cv2.waitKey(0)==ord('q'):
    cv2.destroyAllWindows()
