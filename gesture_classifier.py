from google.colab import files
import os
import tensorflow as tf
assert tf.__version__.startswith('2')

from mediapipe_model_maker import gesture_recognizer

import matplotlib.pyplot as plt

def train_gesture_classifier(data_file_path):
    #verify
    print(data_file_path)
    labels = []
    for i in os.listdir(data_file_path):
        if os.path.isdir(os.path.join(data_file_path), i):
            labels.append(i)
    print(labels)
    
    '''
    Pre-packaged hand detection model from MediaPipe Hands to detect the hand landmarks from the images. 
    Any images without detected hands are ommitted from the dataset. 
    The resulting dataset will contain the extracted hand landmark positions from each image, rather than images themselves.
    '''
    data = gesture_recognizer.Dataset.from_folder(dirname= data_file_path, hparams= gesture_recognizer.HandDataPreprocessingParams())
    #split the data into train, val and test sets
    train_data, rest_data = data.split(0.8) #80% for training, 20% for validation and testing
    val_data, test_data = rest_data.split(0.5) #50% of the remaining 20% for validation, 50% for testing
    
    '''
    Train a gesture classifier using the preprocessed dataset.
    '''
    hparams = gesture_recognizer.HParams(export_dir="exported_model")
    options = gesture_recognizer.GestureRecognizerOptions(hparams=hparams)
    model = gesture_recognizer.GestureRecognizer.create(
    train_data=train_data,
    validation_data=val_data,
    options=options
    )
    
    #Evaluate the model
    loss, acc = model.evaluate(test_data, batch_size=1)
    print(f"Test loss:{loss}, Test accuracy:{acc}")
    
    return model
    