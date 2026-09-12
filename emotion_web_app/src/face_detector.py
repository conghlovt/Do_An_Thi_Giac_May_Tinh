import threading

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from src.config import (
    FACE_BOX_PADDING,
    FACE_DETECTOR_MODEL_PATH,
    MIN_DETECTION_CONFIDENCE,
)


if not FACE_DETECTOR_MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Không tìm thấy MediaPipe face detector: {FACE_DETECTOR_MODEL_PATH}"
    )

# MediaPipe 0.10.30+ chỉ expose Tasks API; detector dùng model TFLite cục bộ
# nên app không cần tải model khi khởi động.
_face_detector_options = vision.FaceDetectorOptions(
    base_options=python.BaseOptions(model_asset_path=str(FACE_DETECTOR_MODEL_PATH)),
    min_detection_confidence=MIN_DETECTION_CONFIDENCE,
)
_face_detector = vision.FaceDetector.create_from_options(_face_detector_options)
_detector_lock = threading.Lock()


def _clip(value, min_value, max_value):
    """Giữ tọa độ nằm trong kích thước ảnh."""
    return max(min_value, min(value, max_value))


def _to_rgb_for_mediapipe(image):
    """
    Chuẩn hóa ảnh đầu vào cho MediaPipe.

    App dùng OpenCV nên frame thường là BGR. Nếu ảnh RGB bị truyền vào,
    face detection vẫn thường chạy được vì detector dựa nhiều vào hình dạng.
    """
    if image is None:
        raise ValueError("Ảnh đầu vào đang rỗng.")

    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def detect_faces(image):
    """
    Phát hiện khuôn mặt bằng MediaPipe.

    Args:
        image: ảnh/frame dạng OpenCV BGR hoặc RGB.

    Returns:
        Danh sách bounding box [(x1, y1, x2, y2), ...].
    """
    rgb_image = _to_rgb_for_mediapipe(image)
    height, width = rgb_image.shape[:2]

    media_pipe_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=np.ascontiguousarray(rgb_image),
    )
    with _detector_lock:
        results = _face_detector.detect(media_pipe_image)
    if not results.detections:
        return []

    boxes = []
    for detection in results.detections:
        bounding_box = detection.bounding_box
        x1 = int(bounding_box.origin_x)
        y1 = int(bounding_box.origin_y)
        box_w = int(bounding_box.width)
        box_h = int(bounding_box.height)
        x2 = x1 + box_w
        y2 = y1 + box_h

        padding = int(max(box_w, box_h) * FACE_BOX_PADDING)
        x1 = _clip(x1 - padding, 0, width - 1)
        y1 = _clip(y1 - padding, 0, height - 1)
        x2 = _clip(x2 + padding, 0, width - 1)
        y2 = _clip(y2 + padding, 0, height - 1)

        if x2 > x1 and y2 > y1:
            boxes.append((x1, y1, x2, y2))

    return boxes
