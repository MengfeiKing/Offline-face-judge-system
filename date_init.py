import face_recognition
import numpy as np
image1 = face_recognition.load_image_file(".\DataSpace\KnownPeople\人物1.jpg")
face1 = face_recognition.face_encodings(image1)[0]
# Create arrays of known face encodings and their names
known_face_encodings = [
    face1
]
f = open('./DataSpace/TxtData/known_face_names.txt','w')
f.write("人物1")
f.close()
known_face_encodings=np.array(known_face_encodings)
np.save('./DataSpace/TxtData/known_face_encodings.npy',known_face_encodings)
print(type(known_face_encodings))
print(known_face_encodings.shape)



