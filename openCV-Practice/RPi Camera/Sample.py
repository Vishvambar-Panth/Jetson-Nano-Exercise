import cv2 
dispW=320
dispH=240
flip=2
camset='nvarguscamerasrc sensor_id=0 ! video/x-raw(memory:NVMM), width=3264, height=2464, format=NV12, framerate=21/1 ! nvvidconv flip-method='+str(flip)+' ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'
cam=cv2.VideoCapture(camset)
while True:
    ret,frame=cam.read()
    cv2.imshow('PiCamera',frame)
    cv2.moveWindow('PiCamera',0,0)
    if cv2.waitKey(1)==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()