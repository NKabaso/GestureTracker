import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def main():    
    """
    Hand tracking part
    
    hand_landmarker.task: trained model that can recognize different aspects/parts of a hand
    """
   #Creates a HandLandmarker object
    base_options = python.BaseOptions(model_asset_path = 'hand_landmarker.task')
    options = vision.HandLandmarkerOptions(
        base_options =base_options,
        num_hands =2
    )
    detector= vision.HandLandmarker.create_from_options(options)
    
    #Create a video capture object
    videCap = cv2.VideoCapture(0) # 0 = default camera
    
    while True:
        success, frame = videCap.read()
        if not success: #checks if the frame was successfully read
            print("Failed to capture video")
            break
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convert the frame to RGB format
        mp_image = mp.Image(image_format = mp.ImageFormat.SRGB, data = rgb_frame) # Create a MediaPipe Image object
        detection_result = detector.detect(mp_image) # Perform hand landmark detection
        
        if detection_result.hand_landmarks: # Check if any hand landmarks were detected
            """
            for hand in detection_result.hand_landmarks:
                for landmark in hand:
                    x = int(landmark.x * frame.shape[1]) # Convert normalized x coordinate to pixel value
                    y = int(landmark.y * frame.shape[0]) # Convert normalized y coordinate to pixel value
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1) # Draw a circle at the landmark position
                    """
        annotated_image = draw_detection_visuals(mp_image.numpy_view(), detection_result)
        #cv2.imshow("Hand Tracking", frame) # Display the frame with hand landmarks
        cv2.imshow("Hand Tracking", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(1) & 0xFF == ord('q'): # Exit the loop if 'q' is pressed
            break

#Handles how the detection marks look like
def draw_detection_visuals(image, detection_result):
    MARGIN = 10  # pixels
    FONT_SIZE = 1
    FONT_THICKNESS = 1
    HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green
    
    mp_hands = mp.tasks.vision.HandLandmarksConnections
    mp_drawing = mp.tasks.vision.drawing_utils
    mp_drawing_styles = mp.tasks.vision.drawing_styles
    
    hand_landmarks_list = detection_result.hand_landmarks
    which_hand_list = detection_result.handedness # Handedness represents whether the detected hands are left or right hands.
    annotated_image = np.copy(image)
    
    #Loop through detected hands
    #Revise
    for index in range(len (hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[index]
        handedness = which_hand_list[index]
        
       # Draw the hand landmarks.
        mp_drawing.draw_landmarks(
            annotated_image,
            hand_landmarks, 
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()) 
    
        #Gets the top left corner of the detected hand's bounding box
        height, width, _ = annotated_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN
        
        #Draw hand label
        # The `cv2.putText()` function is used to draw text on an image. In this specific line of
        # code:
        cv2.putText(annotated_image, f"{handedness[0].category_name}", 
                    (text_x, text_y),cv2.FONT_HERSHEY_DUPLEX,
                    FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)
    return annotated_image

main()