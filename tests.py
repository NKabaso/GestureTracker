import cv2
import joblib
import tensorflow as tf
import mediapipe as mp
import numpy as np
import os

from model_maker import extract_hand_landmarks
from model_maker import build_dataset

def test_versions():
    assert tf.__version__.startswith("2.13"), "TensorFlow version should be 2.13.x"
    assert mp.__version__.startswith("0.10"), "MediaPipe version should be 0.10.x"
    assert np.__version__.startswith("1.24"), "NumPy version should be 1.24.x"
    

def test_model_accuracy():
    # Load the trained model
    model_path = "hand_gesture_model.pkl"
    assert os.path.exists(model_path), f"Model file '{model_path}' does not exist"
    
    model = joblib.load(model_path)
    # Load the dataset
    test1X, test1Y = build_dataset("asl_test2")
    test2X, test2Y = build_dataset("asl_test3")
    print ("Accuracy on test set 1:", model.score(test1X.reshape(test1X.shape[0], -1), test1Y))
    print ("Accuracy on test set 2:", model.score(test2X.reshape(test2X.shape[0], -1), test2Y))
    
if __name__ == "__main__":
    test_model_accuracy()
    