import cv2
import mediapipe as mp
import numpy as np
import os

# Setup MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Create folder for storing landmarks
if not os.path.exists("data"):
    os.makedirs("data")

# Labels for ASL letters (you can reduce for testing)
labels = ['A', 'B', 'C']

cap = cv2.VideoCapture(0)
count = 0
label = 'A'  # Change this when collecting data for other letters

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
            landmarks = []
            for lm in handLms.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            
            # Save data
            np.save(f"data/{label}_{count}.npy", np.array(landmarks))
            count += 1

    cv2.putText(frame, f'Collecting {label}: {count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
