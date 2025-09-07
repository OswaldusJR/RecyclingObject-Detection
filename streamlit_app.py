# streamlit_app.py
import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
from av import VideoFrame

st.set_page_config(page_title="YOLOv11(s) Detector", layout="wide")
st.title("YOLOv11(s) Object Detection with Webcam")

# --- Load model ---
MODEL_PATH = "models/best.onnx"
model = YOLO(MODEL_PATH)

# --- Sidebar for detection settings ---
st.sidebar.title("Detection Settings")
confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.01)
overlap_thresh = st.sidebar.slider("Overlap Threshold", 0.0, 1.0, 0.45, 0.01)  # friendly IoU name
object_classes_input = st.sidebar.text_input("Classes to detect (comma separated, leave empty for all)", "")

if object_classes_input:
    object_classes = [cls.strip() for cls in object_classes_input.split(",")]
else:
    object_classes = None

# --- VideoProcessor for live detection ---
class YOLOWebcamProcessor(VideoProcessorBase):
    def __init__(self):
        self.confidence = confidence
        self.overlap = overlap_thresh
        self.object_classes = object_classes

    def recv(self, frame: VideoFrame) -> VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        # Map class names to indices
        cls_indices = None
        if self.object_classes:
            cls_indices = []
            for cls in self.object_classes:
                if cls in model.names:
                    cls_indices.append(list(model.names).index(cls))

        # Run YOLO inference
        results = model(img, conf=self.confidence, iou=self.overlap, classes=cls_indices)

        # Draw detections
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                conf_score = float(box.conf[0])
                label = f"{model.names[cls_id]} {conf_score:.2f}"
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, label, (x1, max(0, y1-5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return VideoFrame.from_ndarray(img, format="bgr24")

# --- Mode selection ---
option = st.selectbox("Choose mode", ["Upload Image/Video", "Live Webcam"])

if option == "Upload Image/Video":
    uploaded_file = st.file_uploader("Upload an image or video", type=["jpg", "jpeg", "png", "mp4", "avi"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        if uploaded_file.name.lower().endswith((".jpg", ".jpeg", ".png")):
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            cls_indices = None
            if object_classes:
                cls_indices = []
                for cls in object_classes:
                    if cls in model.names:
                        cls_indices.append(list(model.names).index(cls))

            results = model(img, conf=confidence, iou=overlap_thresh, classes=cls_indices)

            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls_id = int(box.cls[0])
                    conf_score = float(box.conf[0])
                    label = f"{model.names[cls_id]} {conf_score:.2f}"
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, label, (x1, max(0, y1-5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Detected Image", use_column_width=True)

        else:  # Video
            import tempfile
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            st.video(tfile.name)

elif option == "Live Webcam":
    st.info("Starting live webcam detection...")
    webrtc_streamer(
        key="yolo-webcam",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=YOLOWebcamProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )
