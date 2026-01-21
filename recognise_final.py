import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import os
import time
import math

MODEL_PATH = "mnist_cnn.h5"

# ---------------------------
# STEP 1: Train model if not saved (Improved with Augmentation logic optional)
# ---------------------------
if not os.path.exists(MODEL_PATH):
    print("Training new MNIST model...")
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    model = Sequential([
        Conv2D(32, (3,3), activation="relu", input_shape=(28,28,1)),
        MaxPooling2D((2,2)),
        Conv2D(64, (3,3), activation="relu"),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(10, activation="softmax")
    ])

    model.compile(optimizer=Adam(0.001), loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(x_train, y_train, validation_split=0.1, epochs=5, batch_size=128, verbose=2)
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}")
    model.save(MODEL_PATH)
    print("Model saved as", MODEL_PATH)
else:
    print("Model already exists. Loading...")
    model = load_model(MODEL_PATH)
    print("Loaded model:", model.input_shape)

# ---------------------------
# STEP 2: Helper Functions
# ---------------------------
def preprocess_canvas(canvas, out_size=28):
    # Center and resize the drawing to be similar to MNIST
    _, th = cv2.threshold(canvas, 50, 255, cv2.THRESH_BINARY)
    nz = cv2.findNonZero(th)
    if nz is None:
        return None
    x,y,w,h = cv2.boundingRect(nz)
    digit = th[y:y+h, x:x+w]
    
    # Resize keeping aspect ratio
    scale = (out_size - 4) / max(h, w)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    digit = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Place in center of 28x28 image
    out = np.zeros((out_size, out_size), dtype=np.uint8)
    xoff, yoff = (out_size - new_w)//2, (out_size - new_h)//2
    out[yoff:yoff+new_h, xoff:xoff+new_w] = digit
    
    # Dilate slightly to match MNIST thickness
    out = cv2.dilate(out, np.ones((2,2), np.uint8), iterations=1)
    
    img = out.astype("float32") / 255.0
    img = img.reshape((1, out_size, out_size, 1))
    return img, out



# ---------------------------
# STEP 3: Air Digit Recognition with Pinch
# ---------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

ret, frame = cap.read()
if ret:
    FRAME_H, FRAME_W = frame.shape[:2]
else:
    FRAME_W, FRAME_H = 640, 480 # Fallback

canvas = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
pts = []
last_point = None
DRAW_RADIUS = 15
print("Air Digit Recognizer ready!")
print(" Instructions:")
print(" - Press 'r' to START Drawing (Green)")
print(" - Press 's' to STOP & PREDICT (Red)")
print(" - Press 'c' to CLEAR")
print(" - Press 'q' to QUIT")

is_drawing_mode = False # State control
last_pred_text = ""
last_pred_conf = 0.0

while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1) # Mirror view
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results = hands.process(rgb)
    
    # Check Keyboard Inputs FIRST (to update state)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        canvas[:] = 0
        last_pred_text = ""
        print("Canvas Cleared")
    elif key == ord('r'):
        is_drawing_mode = True
        print("Drawing Started")
    elif key == ord('s'):
        is_drawing_mode = False
        print("Drawing Stopped. Predicting...")
        if np.count_nonzero(canvas) > 50:
            processed = preprocess_canvas(canvas)
            if processed:
                img_input, vis28 = processed
                preds = model.predict(img_input, verbose=0)
                label = np.argmax(preds[0])
                conf = float(np.max(preds[0]))
                last_pred_text = str(label)
                last_pred_conf = conf
                print(f"Prediction: {label} ({conf:.2f})")
                cv2.imshow("Debug Input", cv2.resize(vis28, (140, 140), interpolation=cv2.INTER_NEAREST))

    if results.multi_hand_landmarks:
        # Get landmarks
        lm = results.multi_hand_landmarks[0].landmark
        
        # Track Index Finger Tip (8) ONLY
        x_px, y_px = int(lm[8].x * FRAME_W), int(lm[8].y * FRAME_H)
        
        if is_drawing_mode:
            # DRAW MODE (Green)
            cv2.circle(frame, (x_px, y_px), 10, (0, 255, 0), -1) 
            
            if last_point:
                cv2.line(canvas, last_point, (x_px, y_px), 255, DRAW_RADIUS * 2)
            else:
                cv2.circle(canvas, (x_px, y_px), DRAW_RADIUS, 255, -1)
            last_point = (x_px, y_px)
            
        else:
            # HOVER MODE (Red)
            cv2.circle(frame, (x_px, y_px), 10, (0, 0, 255), -1) 
            last_point = None # Break the line

    else:
        last_point = None

    # ---------------------------
    # Visualization
    # ---------------------------
    mask = canvas > 0
    overlay = frame.copy()
    overlay[mask] = (255, 255, 0) # Cyan draw color
    
    output = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)

    # UI Text
    status_text = "MODE: DRAWING (r)" if is_drawing_mode else "MODE: STOPPED (s)"
    color = (0, 255, 0) if is_drawing_mode else (0, 0, 255)
    cv2.putText(output, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    if last_pred_text:
        cv2.putText(output, f"Pred: {last_pred_text}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        cv2.putText(output, f"Conf: {last_pred_conf*100:.1f}%", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Air Digit Recognizer", output)

cap.release()
cv2.destroyAllWindows()
