import face_recognition
import cv2
print(cv2.__version__)

don_face = face_recognition.load_image_file('/home/jetson/Desktop/openCV-Practice/Face Detection/faceRecognizer/demoImages/known/Donald Trump.jpg')
don_encode = face_recognition.face_encodings(don_face)[0]
nancy_face = face_recognition.load_image_file('/home/jetson/Desktop/openCV-Practice/Face Detection/faceRecognizer/demoImages/known/Nancy Pelosi.jpg')
nancy_encode = face_recognition.face_encodings(nancy_face)[0]


Encodings = [don_encode,nancy_encode]
Names = ['Donald','Nancy']

font = cv2.FONT_HERSHEY_SIMPLEX
test_image = face_recognition.load_image_file('/home/jetson/Desktop/openCV-Practice/Face Detection/faceRecognizer/demoImages/unknown/u11.jpg')

face_loc = face_recognition.face_locations(test_image)
allEncodings = face_recognition.face_encodings(test_image,face_loc)

test_image = cv2.cvtColor(test_image,cv2.COLOR_RGB2BGR)
for (top,right,bottom,left),faceEncoding in zip(face_loc,allEncodings):
    name='Unknown Person'
    matches = face_recognition.compare_faces(Encodings,faceEncoding)
    if True in matches:
        match_index = matches.index(True)
        name = Names[match_index]
    cv2.rectangle(test_image,(left,top),(right,bottom),(0,0,255),2)
    cv2.putText(test_image,name,(left,top-6),font,1,(0,255,255),2)


frame=cv2.resize(test_image, (640,480))
cv2.imshow('Window',frame)
cv2.moveWindow('Window',0,0)

print('Showing Window')
if cv2.waitKey(0)==ord('q'):
    cv2.destroyAllWindows()


