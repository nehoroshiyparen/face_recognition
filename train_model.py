import os
import face_recognition
import pickle

def train_model_with_dataset():
    if not os.path.exists('dataset'):
        print('Нет папки dataset')
        return

    all_encodings = {
        'names': [],
        'encodings': []
    }

    for image in os.listdir('dataset'):
        if image.endswith(('.png', '.jpg', '.jpeg')):
            face_img = face_recognition.load_image_file(f'dataset/{image}')
            face_encodings = face_recognition.face_encodings(face_img)

            if face_encodings:
                face_enc = face_encodings[0]

                person_name = image.split('.')[0].split('-')[1]
                
                all_encodings['names'].append(person_name)
                all_encodings['encodings'].append(face_enc)
                print(f"Лицо добавлено для {person_name} из файла: {image}")
    
    with open('all_encodings.pickle', 'wb') as file:
        file.write(pickle.dumps(all_encodings))
    
    print('[INFO] Файл all_encodings.pickle успешно создан')

def main(): 
    train_model_with_dataset()

if __name__ == '__main__':
    main()
