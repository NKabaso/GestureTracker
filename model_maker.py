import math
import os
import tensorflow as tf
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True)
    
def extract_hand_landmarks(image):
    if image is None:
        return None
    
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Convert the image to RGB format
    results = hands.process(rgb_image) # Process the image to detect hand landmarks
    
    if not results.multi_hand_landmarks:
        return None
    
    hand = results.multi_hand_landmarks[0] # Get the first detected hand
    wrist = hand.landmark[mp_hands.HandLandmark.WRIST]
    landmark_coordinates = [(lm.x, lm.y, lm.z) for lm in hand.landmark] # Extract the (x, y, z) coordinates of each landmark
    
    relative_coordinates = [(x - wrist.x, y - wrist.y, z - wrist.z) for (x, y, z) in landmark_coordinates] # Convert to relative coordinates
    
    # choose a hand size scale, e.g. distance from wrist to middle finger tip
    mx, my, mz = landmark_coordinates[12]  # middle finger tip
    scale = math.sqrt((mx - wrist.x)**2 + (my - wrist.y)**2 + (mz - wrist.z)**2)
    if scale == 0:
        return None

    normalized = [(x / scale, y / scale, z / scale) for x, y, z in relative_coordinates]
    return normalized

def build_dataset(data_file_path):
    if not os.path.exists(data_file_path):
        print(f"Error: The directory '{data_file_path}' does not exist.")
        return None, None
    X =[] #data
    Y = [] #labels
    for label in os.listdir(data_file_path):
        folder_path = os.path.join(data_file_path, label)
        print(f"Processing folder: {folder_path} with label: {label}")
        cnt = 0
        if os.path.isdir(folder_path):
            for image_file in os.listdir(folder_path):
                #if cnt >= 1600: # Limit to 1600 images per class
                   # break
                image_path = os.path.join(folder_path, image_file)
                image = cv2.imread(image_path)
                if image is None:
                    print("FAILED TO LOAD:", image_path)
                    continue

                landmarks = extract_hand_landmarks(image)
                if landmarks is not None:
                    X.append(landmarks)
                    Y.append(label)
                    cnt += 1
        else:
            print(f"Warning: '{folder_path}' is not a directory and will be skipped.")

    return np.array(X), np.array(Y)

def train(X, Y):
    print("Dataset dimensions:", X.shape, Y.shape)
    # Flatten (x, 21, 3) -> (x, 63)
    data = X.reshape(X.shape[0], -1)
    print("New shape:", data.shape)
    X_train, X_test, Y_train, Y_test = train_test_split(data, Y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, Y_train)
    print("Accuracy:", model.score(X_test, Y_test))
    joblib.dump(model, "hand_gesture_model.pkl")

def main():
    file_path = "asl_alphabet_train/asl_alphabet_train" # Path to the dataset
    data, labels = build_dataset(file_path)
    train(data, labels)

if __name__ == "__main__":    
    main()

