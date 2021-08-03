# This program recognize the face in the given image using the builtin face_recognition module
import face_recognition # Module in Python to Recognize and manipulate faces. Which has these functions --> load_image_file, face_locations, batch_face_locations, face_landmarks, face_encodings, compare_faces, face_distance  
import cv2
print(cv2.__version__)
image=face_recognition.load_image_file('/home/rajavel/Desktop/PyProg/FaceRecognizer/demoImages/unknown/u3.jpg')
image=cv2.resize(image,(640,480))
face_locations=face_recognition.face_locations(image) #Returns an array of bounding boxes of human faces in a image
print(face_locations)
image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
for (row1,col1,row2,col2) in face_locations:
    cv2.rectangle(image,(col1,row1),(col2,row2),(0,0,255),2) # Draw the Rectangle over the identified faces
cv2.imshow('myWindow',image)
cv2.moveWindow('myWindow',0,0)
if cv2.waitKey(0)==ord('q'):
    cv2.destroyAllWindows()