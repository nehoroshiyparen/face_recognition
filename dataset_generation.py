import cv2
import os

path = os.path.dirname(os.path.abspath(__file__))
detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

i = 0
offset = 10
name=input('Введите номер пользователя: ')
video = cv2.VideoCapture(0)

while True:
    ret, im = video.read()
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces=detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100))

    if len(faces) == 0:
        print('Лицо не обнаружено, ожидание')
        cv2.imshow('im', im)
        cv2.waitKey(1)
        continue

    for (x, y, w, h) in faces:
        i = i + 1
        face_img = gray[y-offset:y+h+offset,x-offset:x+w+offset]
        if face_img.size > 0:
            cv2.imwrite('dataset/face-'+name+'.'+str(i)+'.jpg', face_img)
            cv2.rectangle(im,(x-offset,y-offset),(x+w+offset,y+h+offset),(225,0,0),2)
            cv2.imshow('im', face_img)
            cv2.waitKey(70)
    if i > 200:
        break

video.release()
cv2.destroyAllWindows()