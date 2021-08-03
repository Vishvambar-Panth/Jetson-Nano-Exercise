# This program performs object detection using the pretained detectNet and display the frame rate in the image
import jetson.inference
import jetson.utils
import time
import cv2
import numpy as np
timeStamp=time.time()
fpsFilt=0

net=jetson.inference.detectNet('ssd-mobilenet-v2',threshold=0.5)
dispW=640 #1280
dispH=480 #720
flip=2
font=cv2.FONT_HERSHEY_SIMPLEX

# Gstreamer code for improvded Raspberry Pi Camera Quality
#camSet='nvarguscamerasrc wbmode=3 tnr-mode=2 tnr-strength=1 ee-mode=2 ee-strength=1 ! video/x-raw(memory:NVMM), width=3264, height=2464, format=NV12, framerate=21/1 ! nvvidconv flip-method='+str(flip)+' ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! videobalance contrast=1.5 brightness=-.2 saturation=1.2 ! appsink'
#cam=cv2.VideoCapture(camSet)
cam=cv2.VideoCapture('/dev/video0')
cam.set(cv2.CAP_PROP_FRAME_WIDTH,dispW) # set the frame width as dispW
cam.set(cv2.CAP_PROP_FRAME_HEIGHT,dispH)# set the frame height as dispH

#cam=jetson.utils.gstCamera(dispW,dispH,'0')
#cam=jetson.utils.gstCamera(dispW,dispH,'/dev/video0')
#display=jetson.utils.glDisplay()

#while display.IsOpen():
while True:
    #img,width,height=cam.CaptureRGBA()
    _,img=cam.read() # Read the frame
    height=img.shape[0] #Get the image height
    width=img.shape[1]  #Get the image width

    frame=cv2.cvtColor(img,cv2.COLOR_RGB2RGBA).astype(np.float32) # Convert the acquired RGB image into RGBA image to process using cuda
    frame=jetson.utils.cudaFromNumpy(frame) # convert the Numpy arrary to Cuda format
    detections=net.Detect(frame,width,height) # Detect objects in an RGBA image and return a list of detections.
    for detect in detections:
        ID=detect.ClassID # Get the class ID of the detected objects
        top=detect.Top # Get the coordinates of the detected objects
        left=detect.Left
        bottom=detect.Bottom 
        right=detect.Right 
        item=net.GetClassDesc(ID) # Get the class description of the detected object
        print(item,top,left,bottom,right) # Print the items and their coordinates
    #display.RenderOnce(img,width,height)
    dt=time.time()-timeStamp #Compute the change in time
    timeStamp=time.time() # Restart the time stamp
    fps=1/dt # compute the frames per sec
    fpsFilt=0.9*fpsFilt+0.1*fps # Apply LPF to control the abrubt variation
    #print(str(round(fps,1))+' fps ')
    cv2.putText(img,str(round(fpsFilt,1))+'fps',(0,30),font,1,(0,0,255),2) # put the frame rate in the image
    cv2.imshow('DetCam',img) # show the image 
    cv2.moveWindow('DetCam',0,0)
    if cv2.waitKey(1)==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()