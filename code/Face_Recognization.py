import cv2
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(r"path4")
faceCasCade= cv2.CascadeClassifier(r"path1")

font = cv2.FONT_HERSHEY_SIMPLEX

id = 1 # face id on which you want to recognize
names = ['','raj']

cam = cv2.VideoCapture(0,cv2.CAP_DSHOW)
cam.set(3,640)
cam.set(4,480)

minW = 0.1*cam.get(3)
minh = 0.1*cam.get(4)

while True :
    ret , img = cam.read()
    converted_image = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    faces  = faceCasCade.detectMultiScale(
        converted_image,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(int(minW),int(minh)),
    )
    for(x,y,z,w) in faces : 
        cv2.rectangle(img , (x,y) , (x+z,y+w) , (255,0,0) , 2) 
        id,accuracy = recognizer.predict(converted_image[y:y+w,x:x+z])
        if accuracy < 100 :    
            id = names[id]
            accuracy = " {0}%".format(round(100-accuracy))
        else : 
            id = "unknown"
            accuracy = " {0}%".format(round(100-accuracy))
        cv2.putText(img,str(id),(x+5,y-5),font,1,(255,255,255),2)
        cv2.putText(img,str(accuracy),(x+25,y-5),font,1,(255,255,255),2)
    cv2.imshow('camera',img)
    k =cv2.waitKey(100) & 0xff 
    if k==27: 
        break # press escape to close the loop
cam.release()
cv2.destroyAllWindows()