# classical_gesture_mouse.py
import cv2
import numpy as np
import pyautogui
import time

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

screen_w, screen_h = pyautogui.size()
cap = cv2.VideoCapture(0)
cap.set(3, 640); cap.set(4, 480)
cam_w, cam_h = 640, 480

SKIN_LOWER = np.array([0, 40, 80], dtype=np.uint8)
SKIN_UPPER = np.array([15, 255, 255], dtype=np.uint8)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

frame_reduction = 120
smooth = 12
prev_x, prev_y = 0, 0
dragging = False
right_clicked = False
last_click_time = 0
click_cooldown = 0.5
click_fired = False
mouse_active = False

# Finger count smoothing buffer
finger_buffer = []
FINGER_BUFFER_SIZE = 7


def get_skin_mask(frame, face_rects):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, SKIN_LOWER, SKIN_UPPER)
    for (fx, fy, fw, fh) in face_rects:
        pad = 20
        x1 = max(0, fx - pad)
        y1 = max(0, fy - pad)
        x2 = min(frame.shape[1], fx + fw + pad)
        y2 = min(frame.shape[0], fy + fh + int(fh * 2.0))
        mask[y1:y2, x1:x2] = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    return mask


def get_hand_contour(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    valid = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 15000:
            continue
        x, y, w, h = cv2.boundingRect(c)
        aspect = w / (h + 1e-6)
        if 0.4 < aspect < 1.6:
            valid.append(c)
    if not valid:
        return None
    return max(valid, key=cv2.contourArea)


def count_fingers(contour):
    hull = cv2.convexHull(contour, returnPoints=False)
    if len(hull) < 3:
        return 0
    try:
        defects = cv2.convexityDefects(contour, hull)
    except:
        return 0
    if defects is None:
        return 0

    finger_count = 0
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i][0]
        start = tuple(contour[s][0])
        end   = tuple(contour[e][0])
        far   = tuple(contour[f][0])

        a = np.linalg.norm(np.array(end)   - np.array(start))
        b = np.linalg.norm(np.array(far)   - np.array(start))
        c = np.linalg.norm(np.array(end)   - np.array(far))
        angle = np.degrees(np.arccos(
            np.clip((b**2 + c**2 - a**2) / (2*b*c + 1e-6), -1, 1)))

        if angle < 80 and d > 20000:
            finger_count += 1

    return min(finger_count + 1, 5)


def get_stable_finger_count(raw):
    finger_buffer.append(raw)
    if len(finger_buffer) > FINGER_BUFFER_SIZE:
        finger_buffer.pop(0)
    return max(set(finger_buffer), key=finger_buffer.count)


def get_fingertip(contour):
    return tuple(contour[contour[:, :, 1].argmin()][0])


def get_centroid(contour):
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return None
    return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))


def is_palm_up(contour, centroid):
    x, y, w, h = cv2.boundingRect(contour)
    if centroid is None:
        return False
    return centroid[1] > y + h * 0.6


cv2.namedWindow("Classical Gesture Mouse", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Classical Gesture Mouse", 640, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)

    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord('q'):
        break
    if key == ord('t') or key == ord('T'):
        mouse_active = not mouse_active
        if dragging:
            pyautogui.mouseUp()
            dragging = False

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1,
                                          minNeighbors=5, minSize=(80, 80))

    for (fx, fy, fw, fh) in faces:
        pad = 20
        x1 = max(0, fx - pad)
        y1 = max(0, fy - pad)
        x2 = min(cam_w, fx + fw + pad)
        y2 = min(cam_h, fy + fh + int(fh * 2.0))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 165, 0), 2)
        cv2.putText(frame, "excluded", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 165, 0), 1)

    roi = frame[frame_reduction:cam_h - frame_reduction,
                frame_reduction:cam_w - frame_reduction]

    roi_faces = []
    for (fx, fy, fw, fh) in faces:
        roi_faces.append((fx - frame_reduction, fy - frame_reduction, fw, fh))

    mask = get_skin_mask(roi, roi_faces)
    contour = get_hand_contour(mask)

    gesture = "none"
    fingers = 0

    if contour is not None:
        offset_contour = contour + np.array([frame_reduction, frame_reduction])
        cv2.drawContours(frame, [offset_contour], -1, (0, 255, 0), 2)
        hull_points = cv2.convexHull(offset_contour)
        cv2.drawContours(frame, [hull_points], -1, (0, 0, 255), 2)

        raw_fingers = count_fingers(contour)
        fingers = get_stable_finger_count(raw_fingers)

        centroid = get_centroid(contour)
        tip = get_fingertip(contour)

        tip_x = tip[0] + frame_reduction
        tip_y = tip[1] + frame_reduction

        if centroid:
            cv2.circle(frame, (centroid[0] + frame_reduction,
                               centroid[1] + frame_reduction), 8, (255, 255, 0), -1)
        cv2.circle(frame, (tip_x, tip_y), 10, (255, 0, 255), -1)

        mapped_x = np.interp(tip_x, (frame_reduction, cam_w - frame_reduction), (0, screen_w))
        mapped_y = np.interp(tip_y, (frame_reduction, cam_h - frame_reduction), (0, screen_h))
        curr_x = prev_x + (mapped_x - prev_x) / smooth
        curr_y = prev_y + (mapped_y - prev_y) / smooth

        if abs(curr_x - prev_x) < 3 and abs(curr_y - prev_y) < 3:
            curr_x, curr_y = prev_x, prev_y

        if fingers == 1:   gesture = "MOVE"
        elif fingers == 2: gesture = "CLICK"
        elif fingers == 3: gesture = "RIGHT CLICK"
        elif fingers == 4: gesture = "DRAG"
        elif fingers == 5:
            gesture = "SCROLL UP" if is_palm_up(contour, centroid) else "SCROLL DOWN"

        if mouse_active:
            if gesture == "MOVE":
                if not dragging:
                    pyautogui.moveTo(curr_x, curr_y)
                    prev_x, prev_y = curr_x, curr_y
                click_fired = False

            elif gesture == "CLICK":
                t = time.time()
                if not click_fired and (t - last_click_time) > click_cooldown:
                    pyautogui.click()
                    last_click_time = t
                    click_fired = True
            else:
                click_fired = False

            if gesture == "RIGHT CLICK":
                if not right_clicked:
                    pyautogui.rightClick()
                    right_clicked = True
            else:
                right_clicked = False

            if gesture == "DRAG":
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

            if gesture in ("SCROLL UP", "SCROLL DOWN"):
                pyautogui.scroll(15 if gesture == "SCROLL UP" else -15)
                time.sleep(0.01)

    status = "MOUSE: ON  (T = OFF)" if mouse_active else "MOUSE: OFF (T = ON)"
    cv2.putText(frame, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0) if mouse_active else (0, 0, 255), 2)
    cv2.putText(frame, f"Fingers: {fingers}  Gesture: {gesture}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    cv2.putText(frame, "ESC/Q = quit", (20, cam_h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    mask_resized = cv2.resize(mask, (160, 120))
    mask_bgr = cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2BGR)
    frame[cam_h-120:cam_h, cam_w-160:cam_w] = mask_bgr

    cv2.imshow("Classical Gesture Mouse", frame)

if dragging:
    pyautogui.mouseUp()
cap.release()
cv2.destroyAllWindows()