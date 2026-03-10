import cv2
import mediapipe as mp
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
            for hand in detection_result.hand_landmarks:
                for landmark in hand:
                    x = int(landmark.x * frame.shape[1]) # Convert normalized x coordinate to pixel value
                    y = int(landmark.y * frame.shape[0]) # Convert normalized y coordinate to pixel value
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1) # Draw a circle at the landmark position
        cv2.imshow("Hand Tracking", frame) # Display the frame with hand landmarks
        if cv2.waitKey(1) & 0xFF == ord('q'): # Exit the loop if 'q' is pressed
            break

main()