# This file is for handling the model and the UI 

from filters import *
from engine import *
import cv2
import mediapipe as mp


#A simple example of noel hat filter
def main():
    # Initialize model
    detector = init_model()

    # Initialize camera
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        #Convert frame to mediapipe usable
        mp_frame = convert_to_mediapipe(frame)
        
        results = detector.detect(mp_frame)
        # Get face landmarks
        for face_landmarks in results.face_landmarks:
            # Apply filters
            x_start, y_start, hat = noel_hat(face_landmarks, frame)
            frame = overlay_png(frame, hat, x_start, y_start)
        
        cv2.imshow("Noel Hat Filter (Press q to exit)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()