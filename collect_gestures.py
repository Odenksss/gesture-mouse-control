# collect_gestures.py
import cv2, mediapipe as mp, numpy as np, csv, time

LABELS = {
    ord('0'): 'move',         # 1 finger pointing up
    ord('1'): 'click',        # 2 fingers up (index + middle)
    ord('2'): 'right_click',  # 3 fingers up (index + middle + ring)
    ord('3'): 'drag',         # 4 fingers up (index + middle + ring + pinky)
    ord('4'): 'scroll_up',    # open hand palm facing up
    ord('5'): 'scroll_down',  # open hand palm facing down
}

mpHands = mp.solutions.hands
mpDraw = mp.solutions.drawing_utils
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)
cam_w, cam_h = 640, 480
cap.set(3, cam_w); cap.set(4, cam_h)
frame_reduction = 100

def extract_features(lm_list):
    wrist = np.array([lm_list[0].x, lm_list[0].y, lm_list[0].z])
    pts = np.array([[l.x, l.y, l.z] for l in lm_list]) - wrist
    scale = np.linalg.norm(pts[9] - pts[0]) + 1e-6
    pts /= scale
    chains = [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16],[17,18,19,20]]
    angles = []
    for chain in chains:
        for i in range(len(chain)-1):
            a = pts[chain[i]]; b = pts[chain[i+1]]
            angles.append(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-6))
    v1 = pts[5]-pts[0]; v2 = pts[17]-pts[0]
    normal = np.cross(v1,v2); normal /= (np.linalg.norm(normal)+1e-6)
    return np.concatenate([pts.flatten(), angles, normal])

current_label = None
sample_count = {}

# Guide shown on screen for each key
GUIDE = {
    '0': '1 finger up = MOVE',
    '1': '2 fingers up = CLICK',
    '2': '3 fingers up = RIGHT CLICK',
    '3': '4 fingers up = DRAG',
    '4': 'open hand palm UP = SCROLL UP',
    '5': 'open hand palm DOWN = SCROLL DOWN',
}

with open('gesture_data.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'): break
        if key in LABELS:
            current_label = LABELS[key]
            if current_label not in sample_count:
                sample_count[current_label] = 0

        # Yellow rectangle
        cv2.rectangle(frame,
                      (frame_reduction, frame_reduction),
                      (cam_w - frame_reduction, cam_h - frame_reduction),
                      (0, 255, 255), 2)

        # Draw hand landmarks
        if res.multi_hand_landmarks:
            for handLms in res.multi_hand_landmarks:
                mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)

                if current_label:
                    feats = extract_features(handLms.landmark)
                    writer.writerow([current_label] + feats.tolist())
                    sample_count[current_label] += 1

        # Status text
        if current_label:
            cv2.putText(frame,
                        f"Recording: {current_label} ({sample_count.get(current_label, 0)} samples)",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Press a key to start recording",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

        # Key guide on screen
        y = 80
        for k, desc in GUIDE.items():
            color = (0, 255, 0) if (current_label and current_label == LABELS.get(ord(k))) else (180, 180, 180)
            cv2.putText(frame, f"{k}: {desc}", (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            y += 22

        cv2.putText(frame, "ESC or Q to stop and save",
                    (20, cam_h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        cv2.imshow("Collector", frame)

cap.release()
cv2.destroyAllWindows()
print("Done! Samples collected:", sample_count)