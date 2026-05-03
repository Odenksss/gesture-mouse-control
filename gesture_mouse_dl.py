# gesture_mouse_dl.py
import cv2, mediapipe as mp, pyautogui, numpy as np, time
import tensorflow as tf

# --- Load model ---
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

# --- State tracking ---
mouse_active = False
dragging = False
right_clicked = False
click_cooldown = 0.5
last_click_time = 0
click_fired = False

# --- Gesture smoothing ---
gesture_buffer = []
BUFFER_SIZE = 5

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

def get_stable_gesture(new_gesture):
    gesture_buffer.append(new_gesture)
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
    if key == 27 or key == ord('q'):
        break
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

            # Dead zone — ignore tiny tremors
            if abs(curr_x - prev_x) < 3 and abs(curr_y - prev_y) < 3:
                curr_x, curr_y = prev_x, prev_y

            cv2.putText(img, f"{gesture} ({conf:.0%})", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
            cv2.circle(img, (ix, iy), 10, (255, 0, 255), cv2.FILLED)

            if not mouse_active:
                continue

            # --- MOVE ---
            if gesture == 'move':
                if not dragging:
                    pyautogui.moveTo(curr_x, curr_y)
                    prev_x, prev_y = curr_x, curr_y
                click_fired = False

            # --- LEFT CLICK ---
            elif gesture == 'click':
                t = time.time()
                if not click_fired and (t - last_click_time) > click_cooldown:
                    pyautogui.click()
                    last_click_time = t
                    click_fired = True
            else:
                click_fired = False

            # --- RIGHT CLICK ---
            if gesture == 'right_click':
                if not right_clicked:
                    pyautogui.rightClick()
                    right_clicked = True
            else:
                right_clicked = False

            # --- DRAG (text selection) ---
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

            # --- SCROLL ---
            if gesture in ('scroll_up', 'scroll_down'):
                pyautogui.scroll(15 if gesture == 'scroll_up' else -15)
                time.sleep(0.01)

    cv2.imshow("DL Gesture Mouse", img)

cap.release()
cv2.destroyAllWindows()