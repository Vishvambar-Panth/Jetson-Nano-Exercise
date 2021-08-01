import face_recognition
import cv2
print(cv2.__version__)
image = face_recognition.load_image_file('/home/jetson/Desktop/openCV-Practice/Face Detection/faceRecognizer/demoImages/unknown/u4.jpg')
face_loc = face_recognition.face_locations(image)
image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
for (row1,col1,row2,col2) in face_loc:
    cv2.rectangle(image,(col1,row1),(col2,row2),(0,0,255),2)
frame=cv2.resize(image, (640,480))
cv2.imshow('Window',frame)
cv2.moveWindow('Window',0,0)

print('Showing Window')
if cv2.waitKey(0)==ord('q'):
    cv2.destroyAllWindows()
