# streamlit_app.py
import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
from av import VideoFrame
import tempfile

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

if object_classes_input:
    object_classes = [cls.strip() for cls in object_classes_input.split(",")]
else:
    object_classes = None


# --- Helper: Run YOLO + Draw ---
def run_inference_and_draw(img, model, confidence, overlap_thresh, object_classes):
    cls_indices = None
    if object_classes:
        cls_indices = [
            list(model.names).index(cls) for cls in object_classes if cls in model.names
        ]

    results = model(img, conf=confidence, iou=overlap_thresh, classes=cls_indices)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            conf_score = float(box.conf[0])
            label = f"{model.names[cls_id]} {conf_score:.2f}"
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                img,
                label,
                (x1, max(0, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
    return img


class YOLOWebcamProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame_count = 0
        self.skip_rate = 2  # run inference every 3rd frame
        self.last_boxes = []  # store last detections

    def recv(self, frame: VideoFrame) -> VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        orig_h, orig_w = img.shape[:2]

        # For faster inference → resized copy
        resized = cv2.resize(img, (640, 480))
        r_h, r_w = resized.shape[:2]

        self.frame_count += 1
        if self.frame_count % self.skip_rate == 0 or not self.last_boxes:
            try:
                results = model(
                    resized,
                    conf=confidence,
                    iou=overlap_thresh,
                    classes=[
                        list(model.names).index(cls)
                        for cls in object_classes if object_classes and cls in model.names
                    ] if object_classes else None,
                )

                boxes = []
                for r in results:
                    for box in r.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cls_id = int(box.cls[0])
                        conf_score = float(box.conf[0])

                        # Scale back to original size
                        scale_x = orig_w / r_w
                        scale_y = orig_h / r_h
                        x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
                        y1, y2 = int(y1 * scale_y), int(y2 * scale_y)

                        boxes.append((x1, y1, x2, y2, cls_id, conf_score))
                self.last_boxes = boxes

            except Exception as e:
                print(f"[YOLO ERROR] {e}")

        # Draw stored boxes on the *original* frame
        for (x1, y1, x2, y2, cls_id, conf_score) in self.last_boxes:
            label = f"{model.names[cls_id]} {conf_score:.2f}"
            font_scale = max(0.5, min(orig_w, orig_h) / 600)
            thickness = max(1, int(min(orig_w, orig_h) / 500))

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), thickness)
            cv2.putText(
                img, label, (x1, max(0, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness
            )

        return VideoFrame.from_ndarray(img, format="bgr24")



# --- Mode selection ---
option = st.selectbox("Choose mode", ["Upload Image/Video", "Live Webcam"])

# --- Upload Image/Video ---
if option == "Upload Image/Video":
    uploaded_file = st.file_uploader(
        "Upload an image or video", type=["jpg", "jpeg", "png", "mp4", "avi"]
    )
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)

        # --- Image ---
        if uploaded_file.name.lower().endswith((".jpg", ".jpeg", ".png")):
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            img = run_inference_and_draw(img, model, confidence, overlap_thresh, object_classes)
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Detected Image", use_column_width=True)

        # --- Video ---
        else:
            # Save temp input video
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(file_bytes)

            cap = cv2.VideoCapture(tfile.name)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out_path = tfile.name.replace(".mp4", "_processed.mp4")
            out = cv2.VideoWriter(
                out_path,
                fourcc,
                cap.get(cv2.CAP_PROP_FPS),
                (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))),
            )

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame = run_inference_and_draw(frame, model, confidence, overlap_thresh, object_classes)
                out.write(frame)

            cap.release()
            out.release()

            st.success("Video processed successfully ✅")
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
