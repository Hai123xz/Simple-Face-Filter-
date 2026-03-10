# Main file for handling the filter task

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def overlay_png(frame, overlay, x, y):
    h, w = overlay.shape[:2]

    # Check boundaries
    if x < 0 or y < 0 or x + w > frame.shape[1] or y + h > frame.shape[0]:
        return frame

    # Split overlay channels
    overlay_rgb = overlay[:, :, :3]
    alpha = overlay[:, :, 3] / 255.0

    # Region of interest
    roi = frame[y:y+h, x:x+w]

    # Blend
    for c in range(3):
        roi[:, :, c] = (alpha * overlay_rgb[:, :, c] +
                        (1 - alpha) * roi[:, :, c])

    frame[y:y+h, x:x+w] = roi
    return frame


def init_model():
    # Initialize model
    base_options = python.BaseOptions(model_asset_path='./models/face_landmarker.task')
    options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=5)
    detector = vision.FaceLandmarker.create_from_options(options)
    return detector

def convert_to_mediapipe(frame):
    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process to turn into mp image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    return mp_image

#Start webcame
# cap = cv2.VideoCapture(0)
# hat_img = cv2.imread("./images/noel_hat.png", cv2.IMREAD_UNCHANGED)
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
    
#     # Convert the BGR image to RGB
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
#     # Process to turn into mp image
#     mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

#     result = detector.detect(mp_image)
    
#     # Draw face landmarks on the frame
#     for face_landmarks in result.face_landmarks:
#         x1 = int(face_landmarks[234].x * frame.shape[1])
#         x2 = int(face_landmarks[454].x * frame.shape[1])
#         face_width = abs(x2 - x1)

#         hat_width = int(face_width * 1.2)
#         hat_height = int(hat_width * hat_img.shape[0] / hat_img.shape[1])
#         hat = cv2.resize(hat_img, (hat_width, hat_height))

#         # forehead landmark
#         forehead = face_landmarks[10]

#         x = int(forehead.x * frame.shape[1])
#         y = int(forehead.y * frame.shape[0])

#         # position hat
#         x_start = x - hat_width // 2
#         y_start = y - hat_height

#         frame = overlay_png(frame, hat, x_start, y_start)
#     #Display
#     cv2.imshow('Face Detection', frame)
    
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()