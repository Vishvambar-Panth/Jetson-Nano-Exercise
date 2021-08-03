# This program uses the saved model "train.pkl" and recognize the face in live video
import face_recognition
import cv2
import os
import pickle
print(cv2.__version__)

Encodings=[]
Names=[]

with open('train.pkl','rb') as f: # open the file "train.pkl" to get back the saved "Encodings" and "Names" of the Training images
    Names=pickle.load(f) 
    Encodings=pickle.load(f)
font=cv2.FONT_HERSHEY_SIMPLEX
cam=cv2.VideoCapture(0)

while True:
    _,frame=cam.read() # read the frame
    frameSmall=cv2.resize(frame,(0,0),fx=0.3,fy=0.3) # to speedup the process resize the acquired frame with the ratio of 1/3
    frameRGB=cv2.cvtColor(frameSmall,cv2.COLOR_RGB2BGR) # Convert RGB to BGR
    facePositions=face_recognition.face_locations(frameRGB,model='cnn') # Find the face_locations in the Test image using cnn model
    allEncodings=face_recognition.face_encodings(frameRGB,facePositions) # Encode the image 
    for (top,right,bottom,left),face_encoding in zip(facePositions,allEncodings): #for all face_positions in the Test image do the following
        name='Unknown Person'
        matches=face_recognition.compare_faces(Encodings,face_encoding) # Compare the encodings of the Test image with the Encodings of the Training image
        if True in matches: # if find the matches
            first_match_index=matches.index(True) # Get the index of the matched face
            name=Names[first_match_index] # Find the name of the matched face
        top=top*3 # These lines simply resize the image back to original size
        right=right*3
        bottom=bottom*3
        left=left*3
        cv2.rectangle(frame,(left,top),(right,bottom),(0,0,255),2) # Draw the Rectangle
        cv2.putText(frame,name,(left,top-6),font,0.75,(0,0,255),2) # Put the name as text
    cv2.imshow('Picture',frame)
    cv2.moveWindow('Picture',0,0)
    if cv2.waitKey(1)==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
