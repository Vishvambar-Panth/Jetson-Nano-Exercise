# # This Program takes Training images from the directory and encode them using face_encoding function from face_recognition and save the encoded images as "train.pkl"
import face_recognition
import cv2
import os
import pickle # Create portable serialized representations of Python objects
print(cv2.__version__)

Encodings=[]
Names=[]

image_dir='/home/rajavel/Desktop/PyProg/FaceRecognizer/demoImages/known' # Training image folder
for root, dirs, files in os.walk(image_dir): # os.walk - Directory tree generator. For each directory in the directory tree rooted at top, yields a 3-tuple - dirpath, dirnames, filenames
    print(files)
    for file in files:
        path=os.path.join(root,file) #Join two or more pathname with file, inserting '/' as needed. 
        print(path)
        name=os.path.splitext(file)[0] # os.path.splitext() method in Python, used to split the path name into a pair root and ext. ext stands for extension of the specified path while root is everything except ext part
        print(name)
        person=face_recognition.load_image_file(path) # Load the Training images from the specified path
        encoding=face_recognition.face_encodings(person)[0] # Encode the person using face_encoding function in face_recognition module
        Encodings.append(encoding) # Append the encodings of all the person with Name
        Names.append(name)
print(Names)

# Open a file named "train.pkl" with handle "f" for writing bytes (wb) and dump the Encodings and Names
with open('train.pkl','wb') as f:
    pickle.dump(Names,f)
    pickle.dump(Encodings,f)
