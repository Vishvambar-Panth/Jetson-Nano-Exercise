# This program performs object detection using the trained detectNet model
import jetson.inference
import jetson.utils
import time
import cv2
timeStamp=time.time()
fpsFilt=0

net=jetson.inference.detectNet('ssd-mobilenet-v2',threshold=0.5) #Object Detection DNN - locates objects in an image
dispW=640 #1280
dispH=480 #720
#cam=jetson.utils.gstCamera(dispW,dispH,'0')
cam=jetson.utils.gstCamera(dispW,dispH,'/dev/video0') # Create a Camera object
display=jetson.utils.glDisplay() # Create a display Object
while display.IsOpen(): # Do until the window is open
    img,width,height=cam.CaptureRGBA() # Capture the image 
    detections=net.Detect(img,width,height) # Detect objects in an RGBA image and return a list of detections
    display.RenderOnce(img,width,height) # Display the image
    dt=time.time()-timeStamp # compute the change in time
    timeStamp=time.time() # restart the current time
    fps=1/dt # compute the frames per second
    fpsFilt=0.9*fpsFilt+0.1*fps # Apply the LPF
    print(str(round(fps,1))+' fps ') 