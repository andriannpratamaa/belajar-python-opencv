import cv2
import mediapipe as mp

# Inisialisasi MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Daftar nama jari (dari ibu jari sampai kelingking)
finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]

# Fungsi untuk mengecek apakah jari terangkat
def fingers_up(hand_landmarks):
    tips = [4, 8, 12, 16, 20]  # ID landmark ujung jari
    fingers = []

    # Cek ibu jari (sumbu X untuk tangan kanan)
    if hand_landmarks.landmark[tips[0]].x < hand_landmarks.landmark[tips[0] - 1].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # Cek jari telunjuk hingga kelingking (sumbu Y)
    for tip in tips[1:]:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers

# Buka webcam
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Balikkan frame (mirror) dan ubah ke RGB
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Proses deteksi tangan
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Gambar landmark tangan
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(255,0,0), thickness=2)
                )

                # Deteksi status jari
                fingers = fingers_up(hand_landmarks)

                # Tampilkan status tiap jari
                for idx, state in enumerate(fingers):
                    text = f"{finger_names[idx]}: {'Up' if state else 'Down'}"
                    cv2.putText(frame, text, (10, 30 + idx * 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Tampilkan jumlah jari terangkat
        if results.multi_hand_landmarks:
            total_up = sum(fingers)
            cv2.putText(frame, f"Fingers Up: {total_up}", (10, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        # Tampilkan frame
        cv2.imshow("MediaPipe Finger Detection", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # Tekan ESC untuk keluar
            break

cap.release()
cv2.destroyAllWindows()
