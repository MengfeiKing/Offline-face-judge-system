import numpy as np
import cv2
import time
import face_recognition
pictureLocation="./DataSpace/KnownPeople/"
video_capture = cv2.VideoCapture(1)
known_face_encodings = np.load('./DataSpace/TxtData/known_face_encodings.npy')
file = open('./DataSpace/TxtData/known_face_names.txt','r')
known_face_names=file.readlines()
file.close()
length=1
print(known_face_encodings.shape)
for n in known_face_encodings.shape:
    if n>0:
        length=n*length
print(known_face_names)
print("Ready")
while True:
    order=input("Input your Order:")
    if order=="exit":
        break
    if order=="add":
        name=input("Input visitor's name:")
        known_face_names.append(name)
        print("Starting to take picture")
        while True:
            ret, frame = video_capture.read()
            if not ret:
                print("Camera Error")
                break
            frame=cv2.flip(frame,1)
            cv2.imshow("Capture",frame)
            if cv2.waitKey(5) & 0xFF == ord('P'):
                time.sleep(1)
                cv2.destroyAllWindows()
                if_sure=input("Add "+name+" to database"+"Are you sure?(Y/N)")
                if if_sure=="N":
                    continue
                cv2.imwrite(pictureLocation+name.strip('\n')+".jpeg",frame,[int(cv2.IMWRITE_JPEG_QUALITY),100])
                known_face_encodings = known_face_encodings.reshape(length, 1)
                known_face_encodings=np.append(known_face_encodings,face_recognition.face_encodings(frame))
                length+=128
                known_face_encodings = known_face_encodings.reshape(int(length / 128),128)
                print("Done")
                break
    if order=="remove":
            name = input("Input name to remove:")+'\n'
            num=0
            for find in known_face_names:
                if find==name:
                    del known_face_names[num]
                    known_face_encodings=np.delete(known_face_encodings,num,axis=0)
                    length-=128
                num+=1
            print("Done")
print(known_face_names)
print(known_face_encodings.shape)
print("Saving.....")
f = open('./DataSpace/TxtData/known_face_names.txt','w')
for nameStep in known_face_names:
    f.write(nameStep.strip('\n')+'\n')
f.close()
known_face_encodings=np.array(known_face_encodings)
np.save('./DataSpace/TxtData/known_face_encodings.npy',known_face_encodings)
print("Finished!")

