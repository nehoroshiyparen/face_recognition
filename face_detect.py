import cv2
from functions import download_model, detect_face

def detect_person():
    known_encodings = download_model('all')
    video = cv2.VideoCapture(0)
    font = cv2.FONT_ITALIC

    while True:
        ret, image = video.read()
        if not ret:
            break

        result = detect_face(image, known_encodings)

        if result and result[0] is not None:
            match, distance, (x, y, w, h) = result
            print(f"Match found: {match}")
            cv2.putText(image, f'{match}', (x + 10, y - 10), font, 1.1, (0, 255, 0), 4)

        cv2.imshow('im', image)
        if cv2.waitKey(20) == 113:
            break

    video.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    detect_person()