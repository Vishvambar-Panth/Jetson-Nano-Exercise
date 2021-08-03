# This program Idetifies the faces in the Test image using the encoded model of Training images
import face_recognition
import cv2
print(cv2.__version__)

donFace=face_recognition.load_image_file('/home/rajavel/Desktop/PyProg/FaceRecognizer/demoImages/known/Donald Trump.jpg')
donEncode=face_recognition.face_encodings(donFace)[0] # Given an image, return the 128-dimension face encoding for each face in the image.

nancyFace=face_recognition.load_image_file('/home/rajavel/Desktop/PyProg/FaceRecognizer/demoImages/known/Nancy Pelosi.jpg')
nancyEncode=face_recognition.face_encodings(nancyFace)[0]

penceFace=face_recognition.load_image_file('/home/rajavel/Desktop/PyProg/FaceRecognizer/demoImages/known/Mike Pence.jpg')
penceEncode=face_recognition.face_encodings(penceFace)[0]

Encodings=[donEncode, nancyEncode, penceEncode] # Create a list named "Encodings" consists of [donEncode, nancyEncode, penceEncode]
Names=['The Donald', 'Nancy Peloci','Mike Pence'] # Create a label for each encodings - Training Images

font=cv2.FONT_HERSHEY_SIMPLEX # define the font style
testImage=face_recognition.load_image_file('/home/rajavel/Desktop/PyProg/FaceRecognizer/demoImages/unknown/u3.jpg') # load the test image
facePositions=face_recognition.face_locations(testImage) # Find the face locations in the test image
allEncodings=face_recognition.face_encodings(testImage,facePositions) # Encoding the Test image using addional argument known as "facePositions"

testImage=cv2.cvtColor(testImage,cv2.COLOR_RGB2BGR) # Convert the Test image from RGB to BGR since openCV will work with BGR format
for (top,right,bottom,left), face_encoding in zip(facePositions,allEncodings): # Face encoding returns (top,right,bottom,left) as its return value
    name='Unknown Person' # Initialize name as unknown at the beggining
    matches=face_recognition.compare_faces(Encodings,face_encoding) # Compare the encoding of the Test image with the encoding of the Training Images. If there is a match it returns True (1) else False (0)
    if True in matches: # If match found
        first_match_index=matches.index(True) # Find the index of the match
        name=Names[first_match_index] # Using the obtained index find the name
    cv2.rectangle(testImage,(left,top),(right,bottom),(0,0,255),2) # Draw the rectangle over the identified face
    cv2.putText(testImage,name,(left,top-6),font,1,(0,255,0),2) # put name using CV2 text form on the identified face
testImage=cv2.resize(testImage,(640,480))
cv2.imshow('myWindow',testImage)
cv2.moveWindow('myWindow',0,0)
if cv2.waitKey(0)==ord('q'):
    cv2.destroyAllWindows()

