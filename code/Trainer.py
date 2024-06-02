import cv2
from PIL import Image
import numpy as np
import os

path = r"path2"

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier(r"path1")

def imgs_and_paths (path) :
    imgpaths = [os.path.join(path,f) for f in os.listdir(path)]
    facesSamples = []
    ids = []

    for imgpath in imgpaths :
        gray_img = Image.open(imgpath).convert('L')
        imgArr = np.array(gray_img,'uint8')
        id = int(os.path.split(imgpath)[-1].split(".")[1])
        faces = detector.detectMultiScale(imgArr)
        for (x,y,z,w) in faces : 
            facesSamples.append(imgArr[y:y+w,x:x+z])
            ids.append(id)
        return facesSamples,ids
    
faces , ids = imgs_and_paths(path)
recognizer.train(faces,np.array(ids))
recognizer.write(r"path3\train.yml")