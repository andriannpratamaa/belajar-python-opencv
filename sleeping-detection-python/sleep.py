import cv2
import mediapipe as mp
import time
import pygame
from gtts import gTTS
import os
import math

# === Generate suara hanya sekali ===
sleep_sound = "mengantuk.mp3"
if not os.path.exists(sleep_sound):
    gTTS("Anda mengantuk, silakan istirahat sejenak", lang="id").save(sleep_sound)

# === Inisialisasi suara ===
pygame.mixer.init()

# === Inisialisasi MediaPipe Face Mesh ===
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# === Video Capture ===
cap = cv2.VideoCapture(1)
cap.set(3, 640)
cap.set(4, 480)

# === Variabel waktu & batas deteksi ===
pTime = 0
cTime = 0
last_play_time = 0
cooldown = 5  
EYE_AR_THRESH = 0.22  
EYE_AR_CONSEC_FRAMES = 10  
closed_eyes_frame = 0

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def euclidean_dist(p1, p2):
    return math.dist(p1, p2)

def eye_aspect_ratio(landmarks, eye_points, img_w, img_h):
    try:
        p1 = (landmarks[eye_points[0]].x * img_w, landmarks[eye_points[0]].y * img_h)
        p2 = (landmarks[eye_points[1]].x * img_w, landmarks[eye_points[1]].y * img_h)
        p3 = (landmarks[eye_points[2]].x * img_w, landmarks[eye_points[2]].y * img_h)
        p4 = (landmarks[eye_points[3]].x * img_w, landmarks[eye_points[3]].y * img_h)
        p5 = (landmarks[eye_points[4]].x * img_w, landmarks[eye_points[4]].y * img_h)
        p6 = (landmarks[eye_points[5]].x * img_w, landmarks[eye_points[5]].y * img_h)

        vertical1 = euclidean_dist(p2, p6)
        vertical2 = euclidean_dist(p3, p5)
        horizontal = euclidean_dist(p1, p4)
        ear = (vertical1 + vertical2) / (2.0 * horizontal)
        return ear
    except:
        return 0.3  

# === Main loop ===
while True:
    success, img = cap.read()
    if not success:
        print("⚠️ Kamera tidak terdeteksi.")
        break
    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)
    img_h, img_w = img.shape[:2]

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            left_ear = eye_aspect_ratio(face_landmarks.landmark, LEFT_EYE, img_w, img_h)
            right_ear = eye_aspect_ratio(face_landmarks.landmark, RIGHT_EYE, img_w, img_h)
            avg_ear = (left_ear + right_ear) / 2.0

            # Deteksi mengantuk
            if avg_ear < EYE_AR_THRESH:
                closed_eyes_frame += 1
            else:
                closed_eyes_frame = 0

            if closed_eyes_frame >= EYE_AR_CONSEC_FRAMES:
                now = time.time()
                if now - last_play_time > cooldown:
                    pygame.mixer.music.load(sleep_sound)
                    pygame.mixer.music.play()
                    last_play_time = now

                cv2.putText(img, "ANDA MENGANTUK!", (200, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
            else:
                cv2.putText(img, "WASPADA", (250, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)
    else:
        cv2.putText(img, "Wajah tidak terdeteksi", (200, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

    # === FPS counter ===
    cTime = time.time()
    fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Deteksi Mengantuk", img)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC untuk keluar
        break

cap.release()
cv2.destroyAllWindows()
