**1-8-2021** :  Face Recog on Nano 

# To Download VSS
curl -L https://github.com/toolboc/vscode/releases/download/1.32.3/code-oss_1.32.3-arm64.deb -o code-oss_1.32.3-arm64.deb

# Camera String (Rpi Cam)
camSet='nvarguscamerasrc !  video/x-raw(memory:NVMM), width=3264, height=2464, format=NV12, framerate=21/1 ! nvvidconv flip-method='+str(flip)+' ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'



**3-8-2021** : Trying Face Recognition using nference



https://teams.microsoft.com/dl/launcher/launcher.html?url=%2F_%23%2Fl%2Fmeetup-join%2F19%3Addfa773a8caf45d49b0d20eb083ffc61%40thread.tacv2%2F1625907850704%3Fcontext%3D%257B%2522Tid%2522%253A%25225beb351c-3fb8-418f-b612-fe36ace96ef3%2522%252C%2522Oid%2522%253A%2522ba4687f6-ac66-4222-bdd3-c6da9dbb8ee2%2522%257D%26anon%3Dtrue&type=meetup-join&deeplinkId=72516465-4e4f-4842-8b7c-36d238887d17&directDl=true&msLaunch=true&enableMobilePage=true&suppressPrompt=true
