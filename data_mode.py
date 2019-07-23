import numpy as np
known_face_encodings = np.load('./DataSpace/TxtData/known_face_encodings.npy')
known_face_encodings = np.array(known_face_encodings)
f = open('./DataSpace/TxtData/known_face_names.txt','r')
known_face_names=f.readlines()
print(known_face_names)
