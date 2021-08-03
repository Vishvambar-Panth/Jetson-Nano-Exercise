# This program find the faces in the Test image using the encoding of Training images. "known" folder includes the images of Training and "unknown" folder includes the images of Test 
import face_recognition
import cv2
import os
print(cv2.__version__)

Encodings=[]
Names=[]

image_dir='/home/rajavel/Desktop/PyProg/FaceRecognizer/demoImages/known' # Image directory of the training image
for root,dirs,files in os.walk(image_dir): # os.walk - Directory tree generator. For each directory in the directory tree rooted at top, yields a 3-tuple - dirpath, dirnames, filenames  
    #print(files)
    for file in files:
        path=os.path.join(root,file) #Join two or more pathname with file, inserting '/' as needed. 
        #print(path)
        name=os.path.splitext(file)[0] # os.path.splitext() method in Python, used to split the path name into a pair root and ext. ext stands for extension of the specified path while root is everything except ext part.
        #print(name)
        person=face_recognition.load_image_file(path) # Load the Training images from the specified path
        encoding=face_recognition.face_encodings(person)[0] # Encode the person using face_encoding function in face_recognition module
        Encodings.append(encoding) # Append the encodings of all the person
        Names.append(name) # Similarly encode the name of the person
print(Names)

font=cv2.FONT_HERSHEY_SIMPLEX # Define the CV2 Font

image_dir='/home/rajavel/Desktop/PyProg/FaceRecognizer/demoImages/unknown' # Image directory of the Test Image
for root,dirs,files in os.walk(image_dir):
    for file in files:
        testImagePath=os.path.join(root,file) 
        testImage=face_recognition.load_image_file(testImagePath)  
        facePositions=face_recognition.face_locations(testImage) # Find the face locations of the Test image
        allEncodings=face_recognition.face_encodings(testImage,facePositions) # Encode the Test images using Test image and face locations
        testImage=cv2.cvtColor(testImage,cv2.COLOR_RGB2BGR) # Convert the color from RGB to BGR
        for (top,right,bottom,left), face_encoding in zip(facePositions,allEncodings): # Get the coodinates of the faces in the Test image
            name='Unknown Person'
            matches=face_recognition.compare_faces(Encodings,face_encoding) # Compare the faces in a Test image with Training images
            if True in matches: # If find matches
                first_match_index=matches.index(True) # Get the match index
                name=Names[first_match_index] # Find the name using match index
            cv2.rectangle(testImage,(left,top),(right,bottom),(0,0,255),2)
            cv2.putText(testImage,name,(left,top-6),font,1,(0,255,255),2)
    #testImage=cv2.resize(testImage,(640,480))
        cv2.imshow('Picture',testImage)
        cv2.moveWindow('Picture',0,0)
        if cv2.waitKey(0)==ord('q'):
            cv2.destroyAllWindows()