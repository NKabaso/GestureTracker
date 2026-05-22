import os
import tensorflow as tf
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from model_maker import extract_hand_landmarks

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True)

def train_gesture_classifier(data_file_path):
    if not os.path.exists(data_file_path):
        raise ValueError("Dataset path does not exist")
    
    labels = []
    for i in os.listdir(data_file_path):
        folder_path = os.path.join(data_file_path, i)
        if os.path.isdir(folder_path):
            labels.append(i)
            
    print("Labels found in dataset:", labels)
    
    '''
    Pre-packaged hand detection model from MediaPipe Hands to detect the hand landmarks from the images. 
    Any images without detected hands are ommitted from the dataset. 
    The resulting dataset will contain the extracted hand landmark positions from each image, rather than images themselves.
    '''
    data = gesture_recognizer.Dataset.from_folder(dirname= data_file_path, hparams= gesture_recognizer.HandDataPreprocessingParams())
    data = data.shuffle(100)
    #split the data into train, val and test sets
    train_data, rest_data = data.split(0.8) #80% for training, 20% for validation and testing
    val_data, test_data = rest_data.split(0.5) #50% of the remaining 20% for validation, 50% for testing
    
    '''
    Train a gesture classifier using the preprocessed dataset.
    '''
    #hparams = gesture_recognizer.HParams(export_dir="exported_model")
    hparams = gesture_recognizer.HParams(
        learning_rate=0.001,
        batch_size=16,
        epochs=20,
        export_dir="exported_model"
    )
    options = gesture_recognizer.GestureRecognizerOptions(hparams=hparams)
    model = gesture_recognizer.GestureRecognizer.create(
    train_data=train_data,
    validation_data=val_data,
    options=options
    )
    
    #Evaluate the model
    loss, acc = model.evaluate(test_data, batch_size=1)
    print(f"Test loss:{loss}, Test accuracy:{acc}")
    
    model.export_model()
    print("Exported model files:", os.listdir("exported_model"))
    print("Model saved at:", os.path.abspath("exported_model/gesture_recognizer.task"))
    

def main():
    cap = cv2.VideoCapture(0)
    file_path = "path_to_your_dataset"
    model = joblib.load("hand_gesture_model.pkl")
    
    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to capture video")
            break
        
        landmarks = extract_hand_landmarks(frame)    
        if landmarks is not None:
            
            landmarks = np.array(landmarks)
            print("Shape before flatten:", landmarks.shape)
            landmarks = landmarks.flatten()
            print("Shape after flatten:", landmarks.shape)
            landmarks = landmarks.reshape(1, -1)
            print("Shape before predict:", landmarks.shape)
            
            pred = model.predict(landmarks)
            cv2.putText(frame, str(pred[0]), (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        
        cv2.imshow('Hand Gesture Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    main()