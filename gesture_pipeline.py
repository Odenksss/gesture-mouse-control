# gesture_pipeline.py
# Combined pipeline: Collect → Train → Run
# Usage:
#   python gesture_pipeline.py collect   — record gesture samples
#   python gesture_pipeline.py train     — train the model
#   python gesture_pipeline.py run       — run the DL gesture mouse
#   python gesture_pipeline.py all       — collect → train → run in sequence

import sys
import os

# ─────────────────────────────────────────────
# SHARED: feature extractor (used by collect + run)
# ─────────────────────────────────────────────
import numpy as np

def extract_features(lm_list):
    wrist = np.array([lm_list[0].x, lm_list[0].y, lm_list[0].z])
    pts = np.array([[l.x, l.y, l.z] for l in lm_list]) - wrist
    scale = np.linalg.norm(pts[9] - pts[0]) + 1e-6
    pts /= scale
    chains = [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16],[17,18,19,20]]
    angles = []
    for chain in chains:
        for i in range(len(chain) - 1):
            a = pts[chain[i]]; b = pts[chain[i+1]]
            angles.append(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6))
    v1 = pts[5] - pts[0]; v2 = pts[17] - pts[0]
    normal = np.cross(v1, v2); normal /= (np.linalg.norm(normal) + 1e-6)
    return np.concatenate([pts.flatten(), angles, normal])


# ─────────────────────────────────────────────
# PHASE 1: COLLECT
# ─────────────────────────────────────────────
def collect():
    import cv2
    import csv
    import mediapipe as mp

    LABELS = {
        ord('0'): 'move',
        ord('1'): 'click',
        ord('2'): 'right_click',
        ord('3'): 'drag',
        ord('4'): 'scroll_up',
        ord('5'): 'scroll_down',
    }
    GUIDE = {
        '0': '1 finger up = MOVE',
        '1': '2 fingers up = CLICK',
        '2': '3 fingers up = RIGHT CLICK',
        '3': '4 fingers up = DRAG',
        '4': 'open hand palm UP = SCROLL UP',
        '5': 'open hand palm DOWN = SCROLL DOWN',
    }

    mpHands = mp.solutions.hands
    mpDraw = mp.solutions.drawing_utils
    hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.5)

    cap = cv2.VideoCapture(0)
    cam_w, cam_h = 640, 480
    cap.set(3, cam_w); cap.set(4, cam_h)
    frame_reduction = 100
    current_label = None
    sample_count = {}

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
                sample_count.setdefault(current_label, 0)

            cv2.rectangle(frame,
                          (frame_reduction, frame_reduction),
                          (cam_w - frame_reduction, cam_h - frame_reduction),
                          (0, 255, 255), 2)

            if res.multi_hand_landmarks:
                for handLms in res.multi_hand_landmarks:
                    mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)
                    if current_label:
                        feats = extract_features(handLms.landmark)
                        writer.writerow([current_label] + feats.tolist())
                        sample_count[current_label] += 1

            if current_label:
                cv2.putText(frame,
                            f"Recording: {current_label} ({sample_count.get(current_label, 0)} samples)",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Press a key to start recording",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

            y = 80
            for k, desc in GUIDE.items():
                color = (0, 255, 0) if (current_label and current_label == LABELS.get(ord(k))) else (180, 180, 180)
                cv2.putText(frame, f"{k}: {desc}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
                y += 22

            cv2.putText(frame, "ESC or Q to stop and save",
                        (20, cam_h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            cv2.imshow("Collector", frame)

    cap.release()
    cv2.destroyAllWindows()
    print("Collection done! Samples:", sample_count)
    return sample_count


# ─────────────────────────────────────────────
# PHASE 2: TRAIN
# ─────────────────────────────────────────────
def train():
    import pandas as pd
    import tensorflow as tf
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    if not os.path.exists('gesture_data.csv'):
        print("ERROR: gesture_data.csv not found. Run 'collect' first.")
        sys.exit(1)

    df = pd.read_csv('gesture_data.csv', header=None)
    if df.empty:
        print("ERROR: gesture_data.csv is empty.")
        sys.exit(1)

    X = df.iloc[:, 1:].values.astype(np.float32)
    le = LabelEncoder()
    y = tf.keras.utils.to_categorical(le.fit_transform(df.iloc[:, 0]))

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(81,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(len(le.classes_), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)]
    )

    model.save('gesture_model.keras')
    np.save('label_classes.npy', le.classes_)
    val_acc = model.evaluate(X_val, y_val, verbose=0)[1]
    print("Classes:", le.classes_)
    print(f"Val accuracy: {val_acc:.2%}")
    return val_acc


# ─────────────────────────────────────────────
# PHASE 3: RUN (DL gesture mouse)
# ─────────────────────────────────────────────
def run():
    import cv2
    import mediapipe as mp
    import pyautogui
    import tensorflow as tf
    import time

    if not os.path.exists('gesture_model.keras') or not os.path.exists('label_classes.npy'):
        print("ERROR: Model not found. Run 'train' first.")
        sys.exit(1)

    model = tf.keras.models.load_model('gesture_model.keras')
    classes = np.load('label_classes.npy', allow_pickle=True)
    CONFIDENCE_THRESHOLD = 0.75

    pyautogui.FAILSAFE = False
    pyautogui.PAUSE = 0
    screen_w, screen_h = pyautogui.size()

    cap = cv2.VideoCapture(0)
    cap.set(3, 640); cap.set(4, 480)
    cam_w, cam_h = 640, 480
    frame_reduction = 100

    mpHands = mp.solutions.hands
    mpDraw = mp.solutions.drawing_utils
    hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7,
                          min_tracking_confidence=0.7)

    prev_x, prev_y = 0, 0
    smooth = 12
    mouse_active = False
    dragging = False
    right_clicked = False
    click_cooldown = 0.5
    last_click_time = 0
    click_fired = False
    gesture_buffer = []
    BUFFER_SIZE = 5

    def get_stable_gesture(g):
        gesture_buffer.append(g)
        if len(gesture_buffer) > BUFFER_SIZE:
            gesture_buffer.pop(0)
        return max(set(gesture_buffer), key=gesture_buffer.count)

    while True:
        success, img = cap.read()
        if not success: break
        img = cv2.flip(img, 1)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'): break
        if key == ord('t') or key == ord('T'):
            mouse_active = not mouse_active
            if dragging:
                pyautogui.mouseUp()
                dragging = False
            gesture_buffer.clear()

        status = "MOUSE: ON  (T = OFF)" if mouse_active else "MOUSE: OFF (T = ON)"
        cv2.putText(img, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, "ESC/Q = quit", (20, cam_h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

                feats = extract_features(handLms.landmark).reshape(1, -1).astype(np.float32)
                probs = model.predict(feats, verbose=0)[0]
                raw_gesture = classes[np.argmax(probs)] if probs.max() >= CONFIDENCE_THRESHOLD else 'none'
                gesture = get_stable_gesture(raw_gesture)
                conf = probs.max()

                lm = handLms.landmark
                ix = int(lm[8].x * cam_w)
                iy = int(lm[8].y * cam_h)
                mapped_x = np.interp(ix, (frame_reduction, cam_w - frame_reduction), (0, screen_w))
                mapped_y = np.interp(iy, (frame_reduction, cam_h - frame_reduction), (0, screen_h))
                curr_x = prev_x + (mapped_x - prev_x) / smooth
                curr_y = prev_y + (mapped_y - prev_y) / smooth

                if abs(curr_x - prev_x) < 3 and abs(curr_y - prev_y) < 3:
                    curr_x, curr_y = prev_x, prev_y

                cv2.putText(img, f"{gesture} ({conf:.0%})", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
                cv2.circle(img, (ix, iy), 10, (255, 0, 255), cv2.FILLED)

                if not mouse_active:
                    continue

                if gesture == 'move':
                    if not dragging:
                        pyautogui.moveTo(curr_x, curr_y)
                        prev_x, prev_y = curr_x, curr_y
                    click_fired = False

                elif gesture == 'click':
                    t = time.time()
                    if not click_fired and (t - last_click_time) > click_cooldown:
                        pyautogui.click()
                        last_click_time = t
                        click_fired = True
                else:
                    click_fired = False

                if gesture == 'right_click':
                    if not right_clicked:
                        pyautogui.rightClick()
                        right_clicked = True
                else:
                    right_clicked = False

                if gesture == 'drag':
                    if not dragging:
                        pyautogui.moveTo(curr_x, curr_y)
                        pyautogui.mouseDown(button='left')
                        dragging = True
                    else:
                        pyautogui.moveTo(curr_x, curr_y)
                    prev_x, prev_y = curr_x, curr_y
                else:
                    if dragging:
                        pyautogui.mouseUp(button='left')
                        dragging = False

                if gesture in ('scroll_up', 'scroll_down'):
                    pyautogui.scroll(15 if gesture == 'scroll_up' else -15)
                    time.sleep(0.01)

        cv2.imshow("DL Gesture Mouse", img)

    cap.release()
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
USAGE = """
Usage:
  python gesture_pipeline.py collect   — record gesture samples to gesture_data.csv
  python gesture_pipeline.py train     — train model from gesture_data.csv
  python gesture_pipeline.py run       — launch DL gesture mouse
  python gesture_pipeline.py all       — collect → train → run in sequence
"""

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(USAGE)
        sys.exit(0)

    mode = sys.argv[1].lower()

    if mode == 'collect':
        collect()
    elif mode == 'train':
        train()
    elif mode == 'run':
        run()
    elif mode == 'all':
        print("=== STEP 1: COLLECT ===")
        collect()
        print("\n=== STEP 2: TRAIN ===")
        train()
        print("\n=== STEP 3: RUN ===")
        run()
    else:
        print(f"Unknown mode: '{mode}'")
        print(USAGE)
        sys.exit(1)
