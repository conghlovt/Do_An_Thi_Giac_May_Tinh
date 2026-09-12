import tempfile
import time
from pathlib import Path

import av
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer

from src.config import CLASS_NAMES_PATH, DEVICE, MODEL_PATH, OUTPUT_DIR
from src.predict import load_class_names, load_emotion_model
from src.video_utils import process_frame, process_video_file


VIDEO_COLUMN_RATIOS = [1, 2, 1]


st.set_page_config(
    page_title="Nhận diện cảm xúc khuôn mặt",
    layout="wide",
)


def render_centered_video(title, video_data):
    """Hiển thị video trong cột giữa, chiếm khoảng nửa chiều rộng trang."""
    _, video_column, _ = st.columns(VIDEO_COLUMN_RATIOS)
    with video_column:
        st.subheader(title)
        st.video(video_data, format="video/mp4", width="stretch")


def create_centered_preview():
    """Tạo placeholder có cùng kích thước với video kết quả."""
    _, preview_column, _ = st.columns(VIDEO_COLUMN_RATIOS)
    with preview_column:
        return st.empty()


@st.cache_resource
def load_resources(model_path):
    """Cache theo đường dẫn để đổi checkpoint sẽ nạp model mới."""
    del model_path  # Tham số được dùng làm cache key.
    class_names = load_class_names()
    model = load_emotion_model()
    return model, class_names


def show_startup_checks():
    """Hiển thị nhanh trạng thái file model, nhãn và thiết bị."""
    st.sidebar.caption(f"Thiết bị: `{DEVICE}`")
    st.sidebar.caption(f"Model: `{MODEL_PATH.name}`")
    st.sidebar.caption(f"Class names: `{CLASS_NAMES_PATH.name}`")

    if not MODEL_PATH.exists():
        st.sidebar.warning("Chưa có file model .pth")

    if not CLASS_NAMES_PATH.exists():
        st.sidebar.warning("Chưa có file class_names.json")


def render_prediction_table(results):
    """Hiển thị kết quả dự đoán từng khuôn mặt."""
    if not results:
        st.warning("Không phát hiện được khuôn mặt")
        return

    rows = [
        {
            "Khuôn mặt": item["face_id"],
            "Cảm xúc": item["label"],
            "Confidence": f"{item['confidence'] * 100:.2f}%",
            "Box": item["box"],
        }
        for item in results
    ]
    st.dataframe(rows, use_container_width=True, hide_index=True)


def image_mode(model, class_names):
    uploaded_file = st.file_uploader(
        "Upload ảnh khuôn mặt",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded_file is None:
        return

    pil_image = Image.open(uploaded_file).convert("RGB")
    rgb_image = np.array(pil_image)
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

    annotated_bgr, results = process_frame(
        bgr_image,
        model,
        class_names,
        return_results=True,
    )
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    col_original, col_result = st.columns(2)
    with col_original:
        st.subheader("Ảnh gốc")
        st.image(rgb_image, use_container_width=True)

    with col_result:
        st.subheader("Kết quả")
        st.image(annotated_rgb, use_container_width=True)

    render_prediction_table(results)


def video_mode(model, class_names):
    uploaded_file = st.file_uploader(
        "Upload video",
        type=["mp4", "avi", "mov"],
    )

    if uploaded_file is None:
        return

    video_bytes = uploaded_file.getvalue()
    render_centered_video("Video gốc", video_bytes)

    if not st.button("Xử lý video", type="primary"):
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = Path(uploaded_file.name).suffix or ".mp4"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_input:
        temp_input.write(video_bytes)
        input_path = Path(temp_input.name)

    output_path = OUTPUT_DIR / f"emotion_result_{int(time.time())}.mp4"
    progress_bar = st.progress(0)
    status_text = st.empty()
    preview_placeholder = create_centered_preview()

    def update_progress(done_frames, total_frames):
        if total_frames > 0:
            ratio = min(done_frames / total_frames, 1.0)
            progress_bar.progress(ratio)
            status_text.text(f"Đã xử lý {done_frames}/{total_frames} frame")
        else:
            status_text.text(f"Đã xử lý {done_frames} frame")

    def update_preview(annotated_frame, done_frames, total_frames):
        annotated_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        frame_status = (
            f"{done_frames}/{total_frames} frame" if total_frames > 0 else f"frame {done_frames}"
        )
        preview_placeholder.image(
            annotated_rgb,
            caption=f"Đang nhận diện trực tiếp — {frame_status}",
            use_container_width=True,
        )

    try:
        result_path, summary = process_video_file(
            input_path=input_path,
            output_path=output_path,
            model=model,
            class_names=class_names,
            progress_callback=update_progress,
            preview_callback=update_preview,
        )
    except Exception as error:
        st.error(f"Không thể xử lý video: {error}")
        return
    finally:
        input_path.unlink(missing_ok=True)

    progress_bar.progress(1.0)
    status_text.text(f"Hoàn tất {summary['processed_frames']} frame")
    preview_placeholder.empty()
    st.success("Đã xử lý xong. Video nhận diện hiển thị ngay bên dưới.")

    result_bytes = result_path.read_bytes()
    render_centered_video("Video kết quả nhận diện", result_bytes)

    metric_columns = st.columns(4)
    metric_columns[0].metric("Frame đã xử lý", summary["processed_frames"])
    metric_columns[1].metric("Frame có khuôn mặt", summary["frames_with_faces"])
    metric_columns[2].metric("Lượt phát hiện", summary["total_face_detections"])
    metric_columns[3].metric("Thời lượng", f"{summary['duration_seconds']:.1f}s")

    if summary["emotion_counts"]:
        emotion_summary = pd.DataFrame(
            [
                {"Cảm xúc": label, "Số lượt": count}
                for label, count in sorted(
                    summary["emotion_counts"].items(), key=lambda item: item[1], reverse=True
                )
            ]
        )
        st.subheader("Thống kê cảm xúc trong video")
        table_column, chart_column = st.columns([1, 2])
        with table_column:
            st.dataframe(emotion_summary, use_container_width=True, hide_index=True)
        with chart_column:
            st.bar_chart(emotion_summary, x="Cảm xúc", y="Số lượt")
    else:
        st.warning("Không phát hiện được khuôn mặt trong video.")

    st.download_button(
        label="Tải video kết quả",
        data=result_bytes,
        file_name=result_path.name,
        mime="video/mp4",
    )


def webcam_mode(model, class_names):
    st.caption("Bấm Start để mở webcam. Bấm Stop trong khung webcam để dừng.")

    class EmotionVideoProcessor(VideoProcessorBase):
        def __init__(self):
            self.model = model
            self.class_names = class_names
            self.previous_time = time.time()

        def recv(self, frame):
            image = frame.to_ndarray(format="bgr24")
            annotated_frame = process_frame(image, self.model, self.class_names)

            now = time.time()
            fps = 1.0 / max(now - self.previous_time, 1e-6)
            self.previous_time = now

            cv2.putText(
                annotated_frame,
                f"FPS: {fps:.1f}",
                (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (40, 220, 120),
                2,
                cv2.LINE_AA,
            )

            return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

    # Giới hạn webcam trong cột giữa để video không kéo rộng toàn bộ trang.
    _, webcam_column, _ = st.columns(VIDEO_COLUMN_RATIOS)
    with webcam_column:
        webrtc_streamer(
            key="emotion-webcam-realtime",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=EmotionVideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
            video_html_attrs={
                "autoPlay": True,
                "controls": True,
                "style": {"width": "100%", "height": "auto"},
            },
            async_processing=True,
        )


def main():
    st.title("Nhận diện cảm xúc khuôn mặt bằng MediaPipe + Model tự train")

    mode = st.sidebar.radio(
        "Chế độ",
        ["Image", "Video", "Webcam realtime"],
    )
    show_startup_checks()

    try:
        model, class_names = load_resources(str(MODEL_PATH.resolve()))
    except Exception as error:
        st.error(str(error))
        st.info(
            "Hãy kiểm tra thư mục models/, file class_names.json và kiến trúc trong src/model.py."
        )
        st.stop()

    st.sidebar.write("Classes:")
    st.sidebar.write(", ".join(class_names))

    if mode == "Image":
        image_mode(model, class_names)
    elif mode == "Video":
        video_mode(model, class_names)
    else:
        webcam_mode(model, class_names)


if __name__ == "__main__":
    main()
