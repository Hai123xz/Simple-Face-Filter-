#This file for containing the necessary code of the filters

import cv2

def noel_hat(face_landmarks, frame):
    hat_img = cv2.imread("./images/noel_hat.png", cv2.IMREAD_UNCHANGED)
    x1 = int(face_landmarks[234].x * frame.shape[1])
    x2 = int(face_landmarks[454].x * frame.shape[1])
    face_width = abs(x2 - x1)

    hat_width = int(face_width * 1.2)
    hat_height = int(hat_width * hat_img.shape[0] / hat_img.shape[1])
    hat = cv2.resize(hat_img, (hat_width, hat_height))

    # forehead landmark
    forehead = face_landmarks[10]

    x = int(forehead.x * frame.shape[1])
    y = int(forehead.y * frame.shape[0])

    # position hat
    x_start = x - hat_width // 2
    y_start = y - hat_height

    return x_start, y_start, hat