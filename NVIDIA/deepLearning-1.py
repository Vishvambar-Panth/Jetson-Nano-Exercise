# This Program performs image classification using the pretrained imagenet
import jetson.inference #Jetson-inference is a training guide for inference on the NVIDIA Jetson Nano
import jetson.utils # C/C++ wrapper Linux utilities for NVIDIA Jetson - camera, codecs, HID, GStreamer, CUDA, OpenGL/XGL
import time
width=640
height=480
cam=jetson.utils.gstCamera(width,height,'/dev/video0') # camera capture object using GStreamer. gstCamera supports both MIPI CSI cameras and V4L2-compliant devices like USB webcams.
#cam=jetson.utils.gstCamera(width,height,'0') # PiCam
display=jetson.utils.glDisplay() #camera display object 
font=jetson.utils.cudaFont() # Define the cuda font
net=jetson.inference.imageNet('googlenet') # Object for net
timeMark=time.time() # Get the current time
fpsFilter=0
while display.IsOpen(): # window will be open until we close the display
    frame,width,height=cam.CaptureRGBA() # Capture the frame 
    classID,confidence=net.Classify(frame,width,height) # classify the capured the image against the trained model/net
    item=net.GetClassDesc(classID) #Get class description as item
    dt=time.time()-timeMark # compute the change in time by subtracting previous time from the current time
    fps=1/dt # compute the frames per second
    fpsFilter=0.95*fpsFilter+0.05*fps # Apply the low pass filter
    timeMark=time.time() # Restart the time
    font.OverlayText(frame,width,height,str(round(fpsFilter,1))+' fps '+item,5,5,font.Magenta,font.Blue) # put text over the recognized item
    display.RenderOnce(frame,width,height) #display the image. This is similar to imshow 
