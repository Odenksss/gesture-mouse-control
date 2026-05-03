# 🖐️ Gesture Mouse Control

Control your mouse using only hand gestures — no keyboard, no mouse needed.  
Built with **MediaPipe**, **TensorFlow**, and **OpenCV**.

---

## ✨ Features

- 🖱️ **Move** the cursor with 1 finger
- 🖱️ **Left click** with 2 fingers
- 🖱️ **Right click** with 3 fingers
- ✊ **Drag** with 4 fingers
- 🖐️ **Scroll up/down** with open palm
- 🤖 **Deep Learning mode** — trained neural network for accurate gesture recognition
- 📷 **Classical mode** — skin segmentation fallback (no model needed)
- 🔄 **Toggle mouse on/off** anytime with `T` key

---

## 📁 Project Structure

```
gesture-mouse-control/
├── gesture_pipeline.py        # Combined pipeline: collect → train → run
├── classical_gesture_mouse.py # Skin segmentation fallback (no ML needed)
├── .gitignore
├── LICENSE
└── README.md
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install opencv-python mediapipe tensorflow pyautogui scikit-learn pandas numpy
```

### 2. Collect gesture samples

```bash
python gesture_pipeline.py collect
```

Hold each gesture in front of your webcam and press the matching key:

| Key | Gesture | Action |
|-----|---------|--------|
| `0` | 1 finger up | Move |
| `1` | 2 fingers up | Left Click |
| `2` | 3 fingers up | Right Click |
| `3` | 4 fingers up | Drag |
| `4` | Open palm facing up | Scroll Up |
| `5` | Open palm facing down | Scroll Down |

Press `ESC` or `Q` when done.

### 3. Train the model

```bash
python gesture_pipeline.py train
```

### 4. Run the gesture mouse

```bash
python gesture_pipeline.py run
```

### Or do everything at once

```bash
python gesture_pipeline.py all
```

---

## 📷 Classical Mode (no ML)

If you don't want to train a model, use the skin-segmentation based controller:

```bash
python classical_gesture_mouse.py
```

> Works out of the box — no data collection or training needed.  
> May be less accurate in varied lighting conditions.

---

## ⌨️ Controls (while running)

| Key | Action |
|-----|--------|
| `T` | Toggle mouse control ON / OFF |
| `ESC` or `Q` | Quit |

---

## ⚙️ How It Works

**Deep Learning Pipeline (`gesture_pipeline.py`)**
1. **Collect** — MediaPipe detects hand landmarks (21 points × 3D coords). Features are extracted: normalized joint positions, inter-joint angles, and palm normal vector → saved to `gesture_data.csv`
2. **Train** — A small neural network (Dense 128 → 64 → softmax) is trained on the collected features
3. **Run** — Live webcam feed is processed, gestures are classified in real time, and mapped to mouse actions via PyAutoGUI

**Classical Pipeline (`classical_gesture_mouse.py`)**  
Uses HSV skin color segmentation + convexity defects to count fingers — no model required.

---

## 📦 Output Files (auto-generated, not in repo)

| File | Description |
|------|-------------|
| `gesture_data.csv` | Raw collected training data |
| `gesture_model.keras` | Trained Keras model |
| `label_classes.npy` | Gesture label encoder |

---

## 💡 Tips

- Collect **at least 200 samples per gesture** for best accuracy
- Use good, consistent lighting when collecting data
- Keep your hand inside the **yellow rectangle** on screen
- The model uses a **5-frame buffer** to smooth out gesture flickering

---

## 🛠️ Requirements

- Python 3.8+
- Webcam
- Windows / macOS / Linux

---

## 📄 License

MIT License — free to use, modify, and distribute.
