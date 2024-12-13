import face_recognition
import pickle
import numpy as np
import cv2

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def download_model(model):
    with open(f'models/{model}_encodings.pickle', 'rb') as file:
        data = pickle.load(file)
    
    return data

def detect_face(image, known_encodings, tolerance = 0.4):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    match = None

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 0), 4)
        face_encodings = face_recognition.face_encodings(image, known_face_locations=[(y, x+w, y+h, x)])

        if face_encodings:
            face_encoding = face_encodings[0]
            distances = face_recognition.face_distance(known_encodings['encodings'], face_encoding)

            min_distance = min(distances)
            min_distance_idx = np.argmin(distances)

            if min_distance <= tolerance:
                match = known_encodings['names'][min_distance_idx]
                print(f'Match found: {match} with distance: {min_distance}')
                return match, min_distance, (x, y, w, h)
            else:
                print('Unknown face')
                return None
    return None