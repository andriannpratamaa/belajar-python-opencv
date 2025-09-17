import cv2
import mediapipe as mp
import time

mp_face_mesh = mp.solutions.face_mesh

# Indeks titik mata
RIGHT_EYE_TOP = 159
RIGHT_EYE_BOTTOM = 145
LEFT_EYE_TOP = 386
LEFT_EYE_BOTTOM = 374

CLOSED_EYE_TIME = 1.5
EYE_AR_THRESH = 0.025

def eye_aspect_ratio(landmarks, top_idx, bottom_idx, h):
    top_y = landmarks.landmark[top_idx].y * h
    bottom_y = landmarks.landmark[bottom_idx].y * h
    return abs(top_y - bottom_y) / h

cap = cv2.VideoCapture(0)
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:
    eye_closed_start = None
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            for landmarks in results.multi_face_landmarks:
                # Gambar titik mata saja
                for idx in [RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM, LEFT_EYE_TOP, LEFT_EYE_BOTTOM]:
                    x = int(landmarks.landmark[idx].x * w)
                    y = int(landmarks.landmark[idx].y * h)
                    cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

                # Hitung EAR rata-rata
                right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM, h)
                left_ear = eye_aspect_ratio(landmarks, LEFT_EYE_TOP, LEFT_EYE_BOTTOM, h)
                avg_ear = (right_ear + left_ear) / 2

                if avg_ear < EYE_AR_THRESH:
                    if eye_closed_start is None:
                        eye_closed_start = time.time()
                    elif time.time() - eye_closed_start >= CLOSED_EYE_TIME:
                        cv2.putText(frame, "ANDA MENGANTUK!", (30, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                else:
                    eye_closed_start = None

        cv2.imshow('Deteksi Kantuk', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
