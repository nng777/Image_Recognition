import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import sys
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

MODEL_PATH = "glasses_detector.keras"
IMG_SIZE = (224, 224)

def load_and_prep(path):
    bgr = cv2.imread(path)
    if bgr is None:
        raise SystemExit(f"Could not read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, IMG_SIZE, interpolation=cv2.INTER_AREA)
    x = np.expand_dims(rgb.astype("float32"), 0)
    return preprocess_input(x)

def main():
    if len(sys.argv) < 2:
        print("predict_image.py /path/to/image.jpg")
        sys.exit(1)
    model = tf.keras.models.load_model(MODEL_PATH)
    x = load_and_prep(sys.argv[1])
    p = float(model.predict(x, verbose=0)[0,0])
    label = "Glasses" if p >= 0.5 else "No Glasses"
    print(f"{label} (score={p:.3f})")

if __name__ == "__main__":
    main()
