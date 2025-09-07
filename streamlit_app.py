# streamlit_app.py
import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
from av import VideoFrame
import tempfile
import time

st.set_page_config(page_title="YOLOv11(s) Detector", layout="wide")
st.title("YOLOv11(s) Object Detection")

# --- Load model ---
MODEL_PATH = "models/best.onnx"
model = YOLO(MODEL_PATH, task="detect")

# --- Sidebar for detection settings ---
st.sidebar.title("Detection Settings")
confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.01)
overlap_thresh = st.sidebar.slider("Overlap Threshold (IoU)", 0.0, 1.0, 0.45, 0.01)
object_classes_input = st.sidebar.text_input(
    "Classes to detect (comma separated, leave empty for all)", ""
)
object_classes = [cls.strip() for cls in object_classes_input.split(",")] if object_classes_input else None


# --- Helper: Run YOLO + Draw ---
def run_inference_and_draw(img, model, confidence, overlap_thresh, object_classes):
    cls_indices = [
        list(model.names).index(cls) for cls in object_classes if cls in model.names
    ] if object_classes else None

    results = model(img, conf=confidence, iou=overlap_thresh, classes=cls_indices)

    height, width = img.shape[:2]
    font_scale = max(0.5, min(width, height) / 800)  # scales with image size
    thickness = max(1, int(min(width, height) / 400))  # scales with image size

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            conf_score = float(box.conf[0])
            label = f"{model.names[cls_id]} {conf_score:.2f}"

            # Draw rectangle
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), thickness)

            # Draw label background for readability
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            cv2.rectangle(img, (x1, max(0, y1 - text_height - baseline)), (x1 + text_width, y1), (0, 255, 0), -1)

            # Draw text over the background
            cv2.putText(
                img, label, (x1, max(0, y1 - baseline)),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness
            )

    return img

# --- Helper: Draw YOLO boxes for live webcam frames ---
def draw_yolo_live(img, model, confidence, overlap_thresh, object_classes):
    """
    Draws YOLO bounding boxes and labels on webcam frames.
    Dynamically scales text and box thickness based on frame size.
    """
    height, width = img.shape[:2]

    # Dynamic scaling for small webcam frames
    font_scale = max(0.5, min(width, height) / 400)
    thickness = max(1, int(min(width, height) / 200))

    # Determine class indices if user specified certain classes
    cls_indices = [
        list(model.names).index(cls) for cls in object_classes if cls in model.names
    ] if object_classes else None

    results = model(img, conf=confidence, iou=overlap_thresh, classes=cls_indices)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            conf_score = float(box.conf[0])
            label = f"{model.names[cls_id]} {conf_score:.2f}"

            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), thickness)

            # Draw label background for readability
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            cv2.rectangle(img, (x1, max(0, y1 - text_height - baseline)),
                          (x1 + text_width, y1), (0, 255, 0), -1)

            # Draw text over the background
            cv2.putText(img, label, (x1, max(0, y1 - baseline)),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)

    return img




# --- Webcam processor with frame skipping ---
class YOLOWebcamProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame_count = 0
        self.skip_rate = 2  # run inference every 2nd frame
        self.last_results = None  # store YOLO results, not entire frame

    def recv(self, frame: VideoFrame) -> VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1

        # Run YOLO inference only every skip_rate frames
        if self.frame_count % self.skip_rate == 0 or self.last_results is None:
            try:
                height, width = img.shape[:2]
                cls_indices = [list(model.names).index(cls) for cls in object_classes if cls in model.names] if object_classes else None
                self.last_results = model(img, conf=confidence, iou=overlap_thresh, classes=cls_indices)
            except Exception as e:
                print(f"[YOLO ERROR] {e}")
                self.last_results = None

        # Draw boxes and labels every frame using the latest results
        if self.last_results:
            height, width = img.shape[:2]
            font_scale = max(0.5, min(width, height) / 400)
            thickness = max(1, int(min(width, height) / 200))
            for r in self.last_results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls_id = int(box.cls[0])
                    conf_score = float(box.conf[0])
                    label = f"{model.names[cls_id]} {conf_score:.2f}"

                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), thickness)
                    (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                    cv2.rectangle(img, (x1, max(0, y1 - text_height - baseline)),
                                  (x1 + text_width, y1), (0, 255, 0), -1)
                    cv2.putText(img, label, (x1, max(0, y1 - baseline)),
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)

        return VideoFrame.from_ndarray(img, format="bgr24")


# --- Mode selection ---
option = st.selectbox("Choose mode", ["Upload Image", "Live Webcam"])

# --- Upload Image ---
if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img = run_inference_and_draw(img, model, confidence, overlap_thresh, object_classes)
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Detected Image", use_container_width=True)

# --- Live Webcam ---
elif option == "Live Webcam":
    st.info("Starting live webcam detection...")
    webrtc_streamer(
        key="yolo-webcam",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=YOLOWebcamProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
