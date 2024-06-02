import cv2
import cv2.data

cam = cv2.VideoCapture(0,cv2.CAP_DSHOW)

cam.set(3,640)
cam.set(4,480)

detector = cv2.CascadeClassifier(r"path1")
faceId = int(input("Enter a User Id : "))

print("Taking the User Face ....")
print("Look at the camera")
count = 0
while True : 
    ret , img = cam.read()
    converted_image = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(converted_image,1.3,5)
    for (x,y,z,w) in faces : 
        cv2.rectangle(img , (x,y) , (x+z,y+w) , (255,0,0) , 2) 
        count += 1
        cv2.imwrite(r"path2\face."+str(faceId)+"."+str(count)+".jpg",converted_image[y:y+w,x:x+z])
        cv2.imshow("image",img)
    k = cv2.waitKey(100) & 0xff 
    if k == 27 :
        break
    elif count >= 50 :  # please take more as possible for more accuracy here i am taking 50
        break
print("Closing the camera")
cam.release()
cv2.destroyAllWindows()