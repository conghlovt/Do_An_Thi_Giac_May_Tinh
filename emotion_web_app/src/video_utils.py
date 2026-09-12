from collections import Counter
from fractions import Fraction
from pathlib import Path

import av
import cv2

from src.face_detector import detect_faces
from src.predict import predict_emotion


BOX_COLOR = (40, 220, 120)
TEXT_COLOR = (255, 255, 255)
TEXT_BG_COLOR = (30, 30, 30)


def _draw_label(frame, x1, y1, text):
    """Vẽ nền chữ để label dễ đọc trên mọi loại ảnh."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.65
    thickness = 2
    text_size, baseline = cv2.getTextSize(text, font, font_scale, thickness)
    text_w, text_h = text_size

    top = max(0, y1 - text_h - baseline - 8)
    bottom = y1
    right = min(frame.shape[1] - 1, x1 + text_w + 10)

    cv2.rectangle(frame, (x1, top), (right, bottom), TEXT_BG_COLOR, -1)
    cv2.putText(
        frame,
        text,
        (x1 + 5, bottom - baseline - 4),
        font,
        font_scale,
        TEXT_COLOR,
        thickness,
        cv2.LINE_AA,
    )


def process_frame(frame, model, class_names, return_results=False):
    """
    Xử lý một frame: detect mặt, crop, predict cảm xúc và vẽ kết quả.

    Args:
        frame: frame OpenCV BGR.
        model: model PyTorch đã load.
        class_names: list tên class theo đúng thứ tự train.
        return_results: True nếu muốn lấy thêm danh sách kết quả từng mặt.

    Returns:
        Frame đã vẽ kết quả. Nếu return_results=True, trả về (frame, results).
    """
    annotated_frame = frame.copy()
    boxes = detect_faces(frame)
    results = []

    for face_id, (x1, y1, x2, y2) in enumerate(boxes, start=1):
        face_crop = frame[y1:y2, x1:x2]

        if face_crop.size == 0:
            continue

        prediction = predict_emotion(face_crop, model, class_names)
        label = prediction["label"]
        confidence = prediction["confidence"]

        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), BOX_COLOR, 2)
        _draw_label(annotated_frame, x1, y1, f"{label}: {confidence * 100:.1f}%")

        results.append(
            {
                "face_id": face_id,
                "box": (x1, y1, x2, y2),
                "label": label,
                "confidence": confidence,
                "probabilities": prediction["probabilities"],
            }
        )

    if return_results:
        return annotated_frame, results

    return annotated_frame


def process_video_file(
    input_path,
    output_path,
    model,
    class_names,
    progress_callback=None,
    preview_callback=None,
):
    """
    Xử lý video upload và lưu video kết quả.

    progress_callback nhận 2 tham số: số frame đã xử lý và tổng frame.
    preview_callback nhận frame đã gắn nhãn để hiển thị trực tiếp trên web.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError("Không mở được video. Hãy kiểm tra định dạng file upload.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # H.264 + yuv420p phát được trên Chrome/Edge/Firefox. Codec mp4v của
    # OpenCV thường tạo file xem được trên desktop nhưng không phát được trong browser.
    output_width = width - (width % 2)
    output_height = height - (height % 2)
    if output_width <= 0 or output_height <= 0:
        cap.release()
        raise RuntimeError("Video có kích thước không hợp lệ.")

    output_container = av.open(
        str(output_path), mode="w", format="mp4", options={"movflags": "+faststart"}
    )
    output_stream = output_container.add_stream(
        "libx264", rate=Fraction(fps).limit_denominator(1000)
    )
    output_stream.width = output_width
    output_stream.height = output_height
    output_stream.pix_fmt = "yuv420p"
    output_stream.options = {"preset": "veryfast", "crf": "23"}

    frame_index = 0
    frames_with_faces = 0
    total_face_detections = 0
    emotion_counts = Counter()
    preview_stride = max(int(round(fps / 2)), 1)

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            annotated_frame, frame_results = process_frame(
                frame, model, class_names, return_results=True
            )
            encoded_frame = annotated_frame[:output_height, :output_width]
            video_frame = av.VideoFrame.from_ndarray(encoded_frame, format="bgr24")
            for packet in output_stream.encode(video_frame):
                output_container.mux(packet)

            frame_index += 1
            if frame_results:
                frames_with_faces += 1
                total_face_detections += len(frame_results)
                emotion_counts.update(item["label"] for item in frame_results)

            if progress_callback is not None:
                progress_callback(frame_index, total_frames)
            if preview_callback is not None and (
                frame_index == 1 or frame_index % preview_stride == 0
            ):
                preview_callback(encoded_frame, frame_index, total_frames)

        for packet in output_stream.encode():
            output_container.mux(packet)
    except Exception:
        output_container.close()
        output_path.unlink(missing_ok=True)
        raise
    finally:
        cap.release()

    output_container.close()

    if frame_index == 0:
        output_path.unlink(missing_ok=True)
        raise RuntimeError("Video không có frame hợp lệ để xử lý.")

    summary = {
        "processed_frames": frame_index,
        "frames_with_faces": frames_with_faces,
        "total_face_detections": total_face_detections,
        "emotion_counts": dict(emotion_counts),
        "fps": float(fps),
        "duration_seconds": frame_index / fps,
        "width": output_width,
        "height": output_height,
    }
    return output_path, summary
