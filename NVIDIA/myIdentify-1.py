# This is a program to check the working condition of all the modules in Jetson inference and utilities
import jetson.inference
import jetson.utils

cam=jetson.utils.gstCamera(640,480,'/dev/video0') # Object for camera
disp=jetson.utils.glDisplay() # Object for display
font=jetson.utils.cudaFont() # Object for font
net=jetson.inference.imageNet('googlenet') # Initialize the net

while disp.IsOpen(): #do until window is open
    frame,width,height=cam.CaptureRGBA() # Capture the frame
    classID,confident=net.Classify(frame,width,height) #Classifiy the image
    item=net.GetClassDesc(classID) # Get the image description
    font.OverlayText(frame,width,height,item,5,5,font.Magenta,font.Blue)#Write the text over the image
    disp.RenderOnce(frame,width,height)#display the image