import face_recognition
import numpy as np
image1 = face_recognition.load_image_file(".\DataSpace\KnownPeople\金孟非.jpg")
face1 = face_recognition.face_encodings(image1)[0]
image2 = face_recognition.load_image_file(".\DataSpace\KnownPeople\人物1.jpg")
face2 = face_recognition.face_encodings(image2)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    face1,
    face2
]
f = open('./DataSpace/TxtData/known_face_names.txt','w')
f.write("金孟非\n人物1\n")
f.close()
known_face_encodings=np.array(known_face_encodings)
np.save('./DataSpace/TxtData/known_face_encodings.npy',known_face_encodings)



