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

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            conf_score = float(box.conf[0])
            label = f"{model.names[cls_id]} {conf_score:.2f}"
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                img, label, (x1, max(0, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )
    return img


# --- Webcam processor with frame skipping ---
class YOLOWebcamProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame_count = 0
        self.skip_rate = 2  # run inference every 2nd frame
        self.last_boxes = []

    def recv(self, frame: VideoFrame) -> VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        orig_h, orig_w = img.shape[:2]

        # Run inference only every skip_rate frames
        self.frame_count += 1
        if self.frame_count % self.skip_rate == 0 or not self.last_boxes:
            try:
                processed = run_inference_and_draw(img.copy(), model, confidence, overlap_thresh, object_classes)
                self.last_boxes = processed
            except Exception as e:
                print(f"[YOLO ERROR] {e}")

        # Use last detections if skipping
        output = self.last_boxes if isinstance(self.last_boxes, np.ndarray) else img
        return VideoFrame.from_ndarray(output, format="bgr24")


# --- Mode selection ---
option = st.selectbox("Choose mode", ["Upload Image", "Upload Video", "Live Webcam"])

# --- Upload Image ---
if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img = run_inference_and_draw(img, model, confidence, overlap_thresh, object_classes)
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Detected Image", use_container_width=True)

# --- Upload Video ---
elif option == "Upload Video":
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())

        cap = cv2.VideoCapture(tfile.name)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_path = tfile.name.replace(".mp4", "_processed.mp4")
        out = cv2.VideoWriter(
            out_path,
            fourcc,
            cap.get(cv2.CAP_PROP_FPS),
            (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))),
        )

        st.info("⏳ Processing video, please wait...")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = run_inference_and_draw(frame, model, confidence, overlap_thresh, object_classes)
            out.write(frame)

        cap.release()
        out.release()

        st.success("✅ Video processed successfully!")
        st.video(out_path)

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
