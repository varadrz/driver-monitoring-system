import cv2
import mediapipe as mp
import numpy as np
import pygame
from utils import eye_aspect_ratio
import time

# -------------------------------
# Initialize MediaPipe
# -------------------------------
mp_face_mesh = mp.solutions.face_mesh
mp_selfie = mp.solutions.selfie_segmentation
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)
selfie_seg = mp_selfie.SelfieSegmentation(model_selection=1)

# -------------------------------
# Initialize Pygame for sounds
# -------------------------------
pygame.init()
horn_sound = pygame.mixer.Sound(r"assets\horn.mp3")
alert_sound = pygame.mixer.Sound(r"assets\warning.mp3")

# -------------------------------
# Parameters
# -------------------------------
cap = cv2.VideoCapture(0)
EAR_THRESHOLD = 0.27          # Adjust if needed
YELLOW_DURATION = 2.0         # Seconds for yellow warning
RED_DURATION = 10.0           # Seconds for red alert
HEAD_PITCH_THRESHOLD = 15.0   # Degrees, head tilting forward

speed = 60.0
target_speed = 60.0
SPEED_CHANGE_RATE = 0.5

eye_closed_start = None
status = "GREEN"
prev_status = "GREEN"

# Load background
bg_img = cv2.imread("assets/car_interior.jpeg")

# 3D model points for head pose estimation
model_points = np.array([
    (0.0, 0.0, 0.0),             # Nose tip
    (0.0, -330.0, -65.0),        # Chin
    (-225.0, 170.0, -135.0),     # Left eye left corner
    (225.0, 170.0, -135.0),      # Right eye right corner
    (-150.0, -150.0, -125.0),    # Left Mouth corner
    (150.0, -150.0, -125.0)      # Right mouth corner
])

# -------------------------------
# Main Loop
# -------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # -------------------------------
    # Selfie Segmentation
    # -------------------------------
    seg_results = selfie_seg.process(frame_rgb)
    mask = seg_results.segmentation_mask
    mask_3ch = cv2.merge([mask, mask, mask])
    bg_resized = cv2.resize(bg_img, (frame.shape[1], frame.shape[0]))
    frame_virtual = (frame * mask_3ch + bg_resized * (1 - mask_3ch)).astype(np.uint8)

    # -------------------------------
    # Face landmarks
    # -------------------------------
    face_results = face_mesh.process(frame_rgb)
    current_time = time.time()

    eye_closed = False
    head_drowsy = False

    if face_results.multi_face_landmarks:
        face_landmarks = face_results.multi_face_landmarks[0]

        # -------------------------------
        # Eye aspect ratio
        # -------------------------------
        left_eye = np.array([[int(face_landmarks.landmark[i].x * frame.shape[1]),
                              int(face_landmarks.landmark[i].y * frame.shape[0])]
                             for i in [33, 160, 158, 133, 153, 144]])
        right_eye = np.array([[int(face_landmarks.landmark[i].x * frame.shape[1]),
                               int(face_landmarks.landmark[i].y * frame.shape[0])]
                              for i in [362, 385, 387, 263, 373, 380]])

        ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
        if ear < EAR_THRESHOLD:
            eye_closed = True

        # -------------------------------
        # Head pose estimation
        # -------------------------------
        image_points = np.array([
            (face_landmarks.landmark[1].x * frame.shape[1], face_landmarks.landmark[1].y * frame.shape[0]),     # Nose tip
            (face_landmarks.landmark[152].x * frame.shape[1], face_landmarks.landmark[152].y * frame.shape[0]), # Chin
            (face_landmarks.landmark[263].x * frame.shape[1], face_landmarks.landmark[263].y * frame.shape[0]), # Left eye corner
            (face_landmarks.landmark[33].x * frame.shape[1], face_landmarks.landmark[33].y * frame.shape[0]),   # Right eye corner
            (face_landmarks.landmark[287].x * frame.shape[1], face_landmarks.landmark[287].y * frame.shape[0]), # Left mouth
            (face_landmarks.landmark[57].x * frame.shape[1], face_landmarks.landmark[57].y * frame.shape[0])    # Right mouth
        ], dtype=np.float64)

        # Camera internals
        focal_length = frame.shape[1]
        center = (frame.shape[1]/2, frame.shape[0]/2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype="double"
        )
        dist_coeffs = np.zeros((4,1))

        success, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        if success:
            rmat, _ = cv2.Rodrigues(rotation_vector)
            pitch = np.arctan2(-rmat[2,1], rmat[2,2]) * 180 / np.pi
            if pitch > HEAD_PITCH_THRESHOLD:
                head_drowsy = True

    # -------------------------------
    # Determine status
    # -------------------------------
    if eye_closed:
        if eye_closed_start is None:
            eye_closed_start = current_time
        elapsed = current_time - eye_closed_start
    else:
        eye_closed_start = None
        elapsed = 0

    if elapsed > RED_DURATION or head_drowsy:
        status = "RED"
    elif elapsed > YELLOW_DURATION or head_drowsy:
        status = "YELLOW"
    else:
        status = "GREEN"

    # -------------------------------
    # Sound triggers
    # -------------------------------
    if status != prev_status:
        if status == "YELLOW":
            pygame.mixer.Sound.play(horn_sound)
        elif status == "RED":
            pygame.mixer.Sound.play(alert_sound)
    prev_status = status

    # -------------------------------
    # Smooth speed animation
    # -------------------------------
    target_speed = 60 if status != "RED" else 0
    if speed < target_speed:
        speed = min(speed + SPEED_CHANGE_RATE, target_speed)
    elif speed > target_speed:
        speed = max(speed - SPEED_CHANGE_RATE, target_speed)

    # -------------------------------
    # Display info
    # -------------------------------
    color_map = {"GREEN": (0,255,0), "YELLOW": (0,255,255), "RED": (0,0,255)}
    cv2.putText(frame_virtual, f"Driver Status: {status}", (50,50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color_map[status], 3)
    cv2.putText(frame_virtual, f"Speed: {int(speed)} km/h", (50,100),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 2)

    cv2.imshow("Driver Monitoring System", frame_virtual)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
pygame.quit()
