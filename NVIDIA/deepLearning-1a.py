'''
This project is an extention of the previous deepLearning-1.py project
here openCV is used to display the image but processed using Jetson inference and utilities
-------------------------------------------------------------------------------------------
'''
import jetson.inference
import jetson.utils
import cv2
import numpy as np
import time
width=640
height=480
cam=jetson.utils.gstCamera(width,height,'/dev/video0')
#cam=jetson.utils.gstCamera(width,height,'0') # PiCam
#display=jetson.utils.glDisplay()
font=jetson.utils.cudaFont()
net=jetson.inference.imageNet('googlenet')
font=cv2.FONT_HERSHEY_SIMPLEX
timeMark=time.time()
fpsFilter=0
while True:
    frame,width,height=cam.CaptureRGBA(zeroCopy=1) #Capture a camera frame and convert it to float4 RGBA
    classID,confidence=net.Classify(frame,width,height) # Classify an RGBA image and return the object's class and confidence.
    item=net.GetClassDesc(classID) # Return the class description for the given object class.
    dt=time.time()-timeMark 
    fps=1/dt
    fpsFilter=0.95*fpsFilter+0.05*fps
    timeMark=time.time()
    frame=jetson.utils.cudaToNumpy(frame,width,height,4) # Create a numpy ndarray wrapping the CUDA memory, without copying it
    frame=cv2.cvtColor(frame,cv2.COLOR_RGBA2BGR).astype(np.uint8)# Convert the RGBA color to BGR to work with CV2 
    cv2.putText(frame,str(round(fpsFilter,1))+' fps '+item,(0,30),font,1,(0,0,255),2)
    cv2.imshow('RecoCam',frame)
    cv2.moveWindow('RecoCam',0,0)
    if cv2.waitKey(1)==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
    
