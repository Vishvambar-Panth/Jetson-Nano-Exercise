# This program find the faces in the Test image using the encoding of Training images. The model of the trained images called "train.pkl" is saved and retained for testing. "known" folder includes the images of Training and "unknown" folder includes the images of Test 
import face_recognition
import cv2
import os
import pickle  # Create portable serialized representations of Python objects.
print(cv2.__version__)

Encodings=[] # Create a empty list to hold face encodings of the Training images
Names=[]   # Create a empty list to hold name of the Training images


image_dir='/home/rajavel/Desktop/PyProg/FaceRecognizer/demoImages/known' # Training image directory
for root, dirs, files in os.walk(image_dir): # os.walk - Directory tree generator. For each directory in the directory tree rooted at top, yields a 3-tuple - dirpath, dirnames, filenames
    print(files)
    for file in files:
        path=os.path.join(root,file) #Join two or more pathname with file, inserting '/' as needed. 
        print(path)
        name=os.path.splitext(file)[0] # os.path.splitext() method in Python, used to split the path name into a pair root and ext. ext stands for extension of the specified path while root is everything except ext part
        print(name)
        person=face_recognition.load_image_file(path) # Load the Training images from the specified path
        encoding=face_recognition.face_encodings(person)[0] # Encode the person using face_encoding function in face_recognition module
        Encodings.append(encoding) # Append the encodings of all the person
        Names.append(name) # Similarly encode the name of the person
print(Names)

# Open a file called "train.pkl" and dump Encodings and Names
with open('train.pkl','wb') as f: #Open a file named "train.pkl" for writing bytes (wb) with handle f using the concept called "context manager"
    pickle.dump(Names,f) #Write a pickled representation of obj (Names) to the open file object file (f).
    pickle.dump(Encodings,f) #Write a pickled representation of obj (Encodings) to the open file object file (f).
Encodings=[] # Empty the Encodins and Names after dump their contents to "train.pkl"
Names=[]

# Open the saved file "train.pkl" for recognition
with open('train.pkl','rb') as f: # open a file "train.pkl" with handle "f" and "rb" read bytes
    Names=pickle.load(f) # Load the saved content back to the original variable Name 
    Encodings=pickle.load(f) # Load the saved content back to the original variable Encoding 

font=cv2.FONT_HERSHEY_SIMPLEX

image_dir='/home/rajavel/Desktop/PyProg/FaceRecognizer/demoImages/unknown' # Image directory of the Test Image
for root,dirs, files in os.walk(image_dir):
    for file in files:
        print(root)
        print(file)
        testImagePath=os.path.join(root,file)
        testImage=face_recognition.load_image_file(testImagePath) # Load the Test image (file)
        facePositions=face_recognition.face_locations(testImage) # Find the face locations of the Test image
        allEncodings=face_recognition.face_encodings(testImage,facePositions) # Encode the Test images using Test image and face locations
        testImage=cv2.cvtColor(testImage,cv2.COLOR_RGB2BGR) # Convert the color from RGB to BGR

        for (top,right,bottom,left),face_encoding in zip(facePositions,allEncodings): # Get the coodinates of the faces in the Test image. "zip" - will pack the facePositions and allEncodings
            name='Unknown Person'
            matches=face_recognition.compare_faces(Encodings,face_encoding) # Compare the faces in a Test image with Training images
            if True in matches: # If find matches
                first_match_index=matches.index(True) # Get the match index
                name=Names[first_match_index] # Find the name using match index
            cv2.rectangle(testImage,(left,top),(right,bottom),(0,0,255),2) # Draw the Rectangle over the identified faces
            cv2.putText(testImage,name,(left,top-6),font,.75,(0,255,255),2) # put the name over the identified faces
        cv2.imshow('Picture', testImage)
        cv2.moveWindow('Picture',0,0)
        if cv2.waitKey(0)==ord('q'):
            cv2.destroyAllWindows()