# Main file for handling the filter task

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Initialize model
base_options = python.BaseOptions(model_asset_path='./models/face_landmarker.task')
options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=5)
detector = vision.FaceLandmarker.create_from_options(options)

#Start webcame
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process to turn into mp image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    result = detector.detect(mp_image)
    
    # Draw face landmarks on the frame
    for face_landmarks in result.face_landmarks:
        for landmark in face_landmarks:
            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

    #Display
    cv2.imshow('Face Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()