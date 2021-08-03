# This program uses the saved model "train.pkl" and recognize the face in live video. In adition to that it shows the Frames rate (frames per second - fps) in the image
import face_recognition
import cv2
import os
import pickle
import time
print(cv2.__version__)

fpsReport=0   # Initialize the fps Report to zero
scaleFactor=0.3 # Scale factor to resize the image
Encodings=[] # Initialize the Encodings and Names to null set
Names=[]

with open('train.pkl','rb') as f: # Open the saved file "train.pkl"
    Names=pickle.load(f) # Reload the Names and Encodings from the file "train.pkl"
    Encodings=pickle.load(f)
font=cv2.FONT_HERSHEY_SIMPLEX

cam= cv2.VideoCapture(0) # Create a object for video capture
timeStamp=time.time() # Get the current time using time module
while True:
    _,frame=cam.read() # Read the frame
    frameSmall=cv2.resize(frame,(0,0),fx=scaleFactor,fy=scaleFactor) # Resize the frame
    frameRGB=cv2.cvtColor(frameSmall,cv2.COLOR_BGR2RGB) # Convert the color to BGR to work with CV2
    facePositions=face_recognition.face_locations(frameRGB,model='cnn') # find the face_positions
    allEncodings=face_recognition.face_encodings(frameRGB,facePositions) #Encode the Test image using face_posions and Image
    for (top,right,bottom,left),face_encoding in zip(facePositions,allEncodings): 
        name='Unkown Person'
        matches=face_recognition.compare_faces(Encodings,face_encoding) # Compare the encodings of training image and test image
        if True in matches:
            first_match_index=matches.index(True) # find the match index
            name=Names[first_match_index] # find the name using the index
        top=int(top/scaleFactor) # Resize the image back to original size by multipling with the inverse of the scale factor
        right=int(right/scaleFactor)
        bottom=int(bottom/scaleFactor)
        left=int(left/scaleFactor)
        cv2.rectangle(frame,(left,top),(right, bottom),(0,0,255),2) # Draw the Rectangle
        cv2.putText(frame,name,(left,top-6),font,.75,(0,0,255),2) # Put the name as text
    dt=time.time()-timeStamp # change in time = current time - previous time before the loop starts
    fps=1/dt # frame rate = 1/change in time
    fpsReport=0.9*fpsReport+0.1*fps # To control the abroubt variation use the low pass filter
    #print('fps is :',round(fpsReport,1))
    timeStamp=time.time() # Restart the time before go back to get the next frame
    cv2.rectangle(frame,(0,0),(100,40),(0,0,255),-1) # Draw the solid Rectangle to show the FPS
    cv2.putText(frame,str(round(fpsReport,1))+'fps',(0,25),font,0.75,(0,255,255),2) # Put the text on top 
    cv2.imshow('Picture',frame)
    cv2.moveWindow('Picture',0,0)
    if cv2.waitKey(1)==ord('q'):
        break

cam.release()
cv2.destroyAllWindows()