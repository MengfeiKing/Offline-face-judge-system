import face_recognition
import cv2
import numpy as np
import time

video_capture = cv2.VideoCapture(1)

known_face_encodings = np.load('./DataSpace/TxtData/known_face_encodings.npy')
known_face_encodings = np.array(known_face_encodings)
f = open('./DataSpace/TxtData/known_face_names.txt','r')
known_face_names=f.readlines()
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
while True:
    ret, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding,tolerance=0.4)
            name = "Unknown"
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
            face_names.append(name)

    process_this_frame = not process_this_frame
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        localtime = time.asctime(time.localtime(time.time()))
        if name != "Unknown":
            name.strip('\n')
            name.strip('\n')
            print(localtime,"  ",name,"已被识别")
        else:
            print(localtime, "  识别到未知人脸")
        time.sleep(1)
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('Q'):
        break
video_capture.release()
cv2.destroyAllWindows()