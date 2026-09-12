from pathlib import Path

import torch


# Thư mục gốc của project emotion_web_app.
BASE_DIR = Path(__file__).resolve().parents[1]

# Cấu hình fallback cho checkpoint Custom ResNet cũ. Checkpoint EfficientNet
# mang sẵn image_size/mean/std và loader sẽ tự động đọc các giá trị đó.
IMG_SIZE = 64
IN_CHANNELS = 3
NORMALIZE_MEAN = (0.5, 0.5, 0.5)
NORMALIZE_STD = (0.5, 0.5, 0.5)

# Đường dẫn model và file nhãn. EfficientNet-B0 VGAF là model mặc định.
MODEL_PATH = BASE_DIR / "models" / "best_efficientnet_b0_vgaf.pth"
CLASS_NAMES_PATH = BASE_DIR / "models" / "class_names.json"
OUTPUT_DIR = BASE_DIR / "runs" / "outputs"

# Khi chạy trực tiếp từ repository, các model dung lượng lớn được đặt trong
# thư mục models/ của project. Vẫn ưu tiên file nằm trong emotion_web_app/models
# để app có thể đóng gói độc lập sau này.
PROJECT_MODELS_DIR = BASE_DIR.parent / "models"
if not MODEL_PATH.exists():
    MODEL_PATH = (
        BASE_DIR.parent
        / "ket_qua_acc_tren_75"
        / "cnn_resnet_emotion"
        / "best_efficientnet_b0_vgaf.pth"
    )

FACE_DETECTOR_MODEL_PATH = BASE_DIR / "models" / "blaze_face_short_range.tflite"
if not FACE_DETECTOR_MODEL_PATH.exists():
    FACE_DETECTOR_MODEL_PATH = PROJECT_MODELS_DIR / "blaze_face_short_range.tflite"

# MediaPipe face detection.
MIN_DETECTION_CONFIDENCE = 0.5
FACE_BOX_PADDING = 0.18

# Tự dùng GPU nếu máy có CUDA, ngược lại chạy CPU.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
