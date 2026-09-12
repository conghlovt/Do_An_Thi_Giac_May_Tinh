import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.config import (
    CLASS_NAMES_PATH,
    DEVICE,
    IMG_SIZE,
    MODEL_PATH,
    NORMALIZE_MEAN,
    NORMALIZE_STD,
)
from src.model import build_model


def load_class_names():
    """Load danh sách class theo đúng thứ tự lúc train."""
    class_path = Path(CLASS_NAMES_PATH)
    if not class_path.exists():
        raise FileNotFoundError("Chưa tìm thấy file nhãn class_names.json")

    with class_path.open("r", encoding="utf-8") as file:
        class_names = json.load(file)

    if not isinstance(class_names, list) or len(class_names) == 0:
        raise ValueError("class_names.json phải là một list tên class và không được rỗng.")

    return class_names


def _extract_state_dict(checkpoint):
    """Hỗ trợ checkpoint dạng raw state_dict hoặc dict có key model_state_dict/state_dict."""
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]

    return checkpoint


def load_emotion_model():
    """Load model và cấu hình preprocessing từ metadata checkpoint."""
    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        raise FileNotFoundError("Chưa tìm thấy model, hãy đặt file model vào thư mục models/")

    class_names = load_class_names()

    try:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        architecture = (
            checkpoint.get("architecture", "custom_resnet")
            if isinstance(checkpoint, dict)
            else "custom_resnet"
        )
        checkpoint_classes = (
            checkpoint.get("class_names") if isinstance(checkpoint, dict) else None
        )
        if checkpoint_classes is not None and list(checkpoint_classes) != class_names:
            raise ValueError(
                "Thứ tự class trong checkpoint không khớp models/class_names.json."
            )

        model = build_model(
            num_classes=len(class_names), architecture=architecture
        )
        state_dict = _extract_state_dict(checkpoint)
        model.load_state_dict(state_dict)
    except (RuntimeError, ValueError, KeyError) as error:
        raise RuntimeError(
            f"Không load được checkpoint {model_path.name}: {error}"
        ) from error

    image_size = (
        checkpoint.get("image_size", checkpoint.get("img_size", IMG_SIZE))
        if isinstance(checkpoint, dict)
        else IMG_SIZE
    )
    normalization_mean = (
        checkpoint.get("normalization_mean", NORMALIZE_MEAN)
        if isinstance(checkpoint, dict)
        else NORMALIZE_MEAN
    )
    normalization_std = (
        checkpoint.get("normalization_std", NORMALIZE_STD)
        if isinstance(checkpoint, dict)
        else NORMALIZE_STD
    )

    # Gắn metadata runtime vào model để predict_emotion luôn tiền xử đúng.
    model.emotion_architecture = architecture
    model.emotion_image_size = int(image_size)
    model.emotion_normalization_mean = tuple(normalization_mean)
    model.emotion_normalization_std = tuple(normalization_std)
    model.emotion_use_grayscale = architecture in {
        "custom_resnet",
        "custom_resnet18",
        None,
    }
    model = model.to(DEVICE)
    model.eval()
    return model


def _opencv_to_pil_rgb(face_image):
    """Chuyển crop khuôn mặt từ OpenCV/numpy sang PIL RGB."""
    if face_image is None:
        raise ValueError("Ảnh khuôn mặt đang rỗng.")

    if isinstance(face_image, Image.Image):
        return face_image.convert("RGB")

    if not isinstance(face_image, np.ndarray):
        raise TypeError("face_image phải là numpy.ndarray hoặc PIL.Image.")

    if face_image.size == 0:
        raise ValueError("Ảnh khuôn mặt có kích thước rỗng.")

    if len(face_image.shape) == 2:
        rgb_image = cv2.cvtColor(face_image, cv2.COLOR_GRAY2RGB)
    elif face_image.shape[2] == 4:
        rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGRA2RGB)
    else:
        rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

    return Image.fromarray(rgb_image).convert("RGB")


def preprocess_face(face_image, model=None):
    """
    Tiền xử lý crop khuôn mặt theo metadata của model.

    EfficientNet dùng RGB 224x224 + ImageNet normalization; CustomResNet cũ
    dùng grayscale 3 kênh 64x64 + normalization (0.5, 0.5, 0.5).
    """
    pil_image = _opencv_to_pil_rgb(face_image)

    image_size = getattr(model, "emotion_image_size", IMG_SIZE)
    normalization_mean = getattr(
        model, "emotion_normalization_mean", NORMALIZE_MEAN
    )
    normalization_std = getattr(model, "emotion_normalization_std", NORMALIZE_STD)
    use_grayscale = getattr(model, "emotion_use_grayscale", True)

    transform_steps = []
    if use_grayscale:
        transform_steps.append(transforms.Grayscale(num_output_channels=3))
    transform_steps.extend(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=normalization_mean, std=normalization_std),
        ]
    )
    transform = transforms.Compose(transform_steps)

    tensor = transform(pil_image)
    return tensor.unsqueeze(0)


def predict_emotion(face_image, model, class_names):
    """
    Dự đoán cảm xúc cho một crop khuôn mặt.

    Returns:
        dict gồm label, confidence và xác suất từng class.
    """
    input_tensor = preprocess_face(face_image, model=model).to(DEVICE)

    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()

    predicted_index = int(np.argmax(probabilities))
    label = class_names[predicted_index]
    confidence = float(probabilities[predicted_index])

    return {
        "label": label,
        "confidence": confidence,
        "probabilities": {
            class_name: float(probabilities[index])
            for index, class_name in enumerate(class_names)
        },
    }
