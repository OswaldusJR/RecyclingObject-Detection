# streamlit_app.py
import streamlit as st
import sys, subprocess, os

st.title("YOLO + OpenCV Test App 🚀")

# --- Force headless OpenCV ---
try:
    import cv2
    if "opencv-python" in cv2.__file__:
        raise ImportError("Wrong OpenCV variant loaded")
except Exception:
    st.warning("⚠️ Fixing OpenCV installation to headless...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--force-reinstall", "opencv-python-headless==4.8.1.78"])
    import cv2

st.success(f"✅ OpenCV loaded from: {cv2.__file__}")

# --- Test Ultralytics YOLO import ---
try:
    from ultralytics import YOLO
    st.success("✅ Ultralytics YOLO imported successfully")
    # Just load a default model to check weights
    model = YOLO("yolov8n.pt")
    st.success("✅ YOLO model loaded (yolov8n.pt)")
except Exception as e:
    st.error(f"❌ YOLO import failed: {e}")

# --- Test a simple OpenCV operation ---
import numpy as np
img = np.zeros((100, 100, 3), dtype=np.uint8)
cv2.putText(img, "YOLO", (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

st.image(img, caption="OpenCV Test Image", channels="BGR")
