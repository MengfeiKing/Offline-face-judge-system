import face_recognition as faceReg
import cv2 as cv
import math
import numpy as np


def cv_imread(image_path):
    cv_img = cv.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
    return cv_img


camera = cv.VideoCapture(0)
known_face_encodings = [
    faceReg.face_encodings(faceReg.load_image_file(".\DataSpace\KnownPeople\金孟非.jpg"))[0],
    faceReg.face_encodings(faceReg.load_image_file(".\DataSpace\KnownPeople\金孟非.jpg"))[0]
]
known_face_names = [
    "Mengfei King",
    "People1"
]

while True:
    ret, src = camera.read()
    src = cv.flip(src, 1)
    OutSrc=src
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    face_judge = cv.CascadeClassifier(".\Package\lbpcascade_frontalface_improved.xml")
    face_find_result = face_judge.detectMultiScale(gray, 1.02, 2)
    FlagFindFace=False
    for x, y, dx, dy in face_find_result:
        FlagFindFace=True
        OutX1=max(0,x-int(0.5*dx))
        OutX2=x+int(1.5*dx)
        OutY1=max(0,y-int(0.8*dx))
        OutY2=y+int(1.3*dy)
        cropImg = src[OutY1:OutY2,OutX1:OutX2]
        faceInput=cropImg[:, :, ::-1]
        faceEncodings = faceReg.face_encodings(faceInput)
        matches = faceReg.compare_faces(known_face_encodings,faceEncodings)
        name = "Unknown"
        if True in matches:
             first_match_index = matches.index(True)
             name = known_face_names[first_match_index]
        print(name," is appearing")
        cv.rectangle(OutSrc, (OutX1, OutY1), (OutX2, OutY2), (0, 255, 255), 3)

    cv.imshow("result", OutSrc)
    if FlagFindFace:
        cv.imshow("face",cropImg)
        cv.waitKey(1)
    if cv.waitKey(1)==27:
        break
