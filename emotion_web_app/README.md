# Emotion Web App

Web app nhận diện cảm xúc khuôn mặt bằng MediaPipe và model PyTorch đã train.
Mặc định app dùng EfficientNet-B0 VGAF; loader vẫn hỗ trợ checkpoint CustomResNet cũ.
MediaPipe chỉ dùng để phát hiện và cắt vùng khuôn mặt.

## Cấu trúc thư mục

```text
emotion_web_app/
├── models/
│   ├── best_efficientnet_b0_vgaf.pth
│   └── class_names.json
├── src/
│   ├── model.py
│   ├── config.py
│   ├── face_detector.py
│   ├── predict.py
│   └── video_utils.py
├── app.py
├── requirements.txt
├── README.md
└── runs/
    └── outputs/
```

## Chọn model

Model mặc định:

```text
models/best_efficientnet_b0_vgaf.pth
```

Muốn dùng checkpoint khác, hãy sửa biến `MODEL_PATH` trong `src/config.py`.
Loader đọc `architecture`, `image_size`, `normalization_mean` và `normalization_std`
từ checkpoint để tự dựng đúng model và tiền xử lý ảnh.

## File class_names.json

File `models/class_names.json` phải chứa danh sách class theo đúng thứ tự lúc train. Ví dụ hiện tại:

```json
[
  "angry",
  "contempt",
  "disgust",
  "fear",
  "happy",
  "neutral",
  "sad",
  "surprise"
]
```

Lưu ý rất quan trọng: nếu thứ tự class khác thứ tự lúc train, model vẫn chạy nhưng sẽ trả nhãn sai.

## Cài thư viện

Mở terminal trong thư mục `emotion_web_app`, sau đó chạy:

```bash
pip install -r requirements.txt
```

## Chạy web

```bash
streamlit run app.py
```

Sau khi chạy, Streamlit sẽ in ra địa chỉ local, thường là:

```text
http://localhost:8501
```

## Cách sử dụng

Chọn chế độ ở sidebar:

- `Image`: upload ảnh `.jpg`, `.jpeg`, `.png`, app sẽ vẽ bounding box và nhãn cảm xúc.
- `Video`: upload video `.mp4`, `.avi`, `.mov`; app hiển thị frame nhận diện trong lúc xử lý,
  phát video kết quả H.264 ngay trên web, thống kê số lượt cảm xúc và cho phép tải file.
  Video kết quả vẫn được lưu trong `runs/outputs/`.
- `Webcam realtime`: mở webcam bằng `streamlit-webrtc`, nhận diện theo thời gian thực và hiển thị FPS.

## Lỗi thường gặp

- Không tìm thấy model `.pth`: kiểm tra `MODEL_PATH` trong `src/config.py`.
- Không tìm thấy `class_names.json`: kiểm tra file `models/class_names.json`.
- Sai thứ tự class: sửa `class_names.json` cho đúng thứ tự class lúc train.
- Sai kiến trúc model: kiểm tra `src/model.py` phải giống kiến trúc lúc train, đặc biệt số layer, tên layer và số class.
- MediaPipe không phát hiện mặt: thử ảnh rõ hơn, mặt nhìn thẳng hơn, đủ sáng hơn.
- Webcam không mở được: kiểm tra quyền camera của trình duyệt, đóng app khác đang dùng camera, thử chạy trên Chrome/Edge.
- CUDA không khả dụng: app sẽ tự chạy CPU, chỉ chậm hơn GPU.

## Ghi chú kiến trúc

File `src/model.py` hỗ trợ hai kiến trúc: EfficientNet-B0 VGAF từ `hsemotion`
và `CustomResNet` tự xây dựng. Kiến trúc được chọn tự động bằng trường
`architecture` trong checkpoint.
