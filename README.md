# Đồ án XLA – nhận diện mã QR

Chương trình đọc mã QR/Data Matrix từ một ảnh hoặc toàn bộ ảnh trong thư mục. Pipeline kết hợp WeChatQRCode, OpenCV QRCodeDetector và ZXing; `pyzbar`/`pylibdmtx` là thư viện tuỳ chọn.

## Cài đặt

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Bốn model WeChatQR đã được giữ trong `wechat_models/`. Nếu thiếu model, chương trình sẽ tự tải khi có mạng.

## Chạy

Chạy không mở cửa sổ OpenCV, phù hợp khi xử lý hàng loạt hoặc chạy trên máy chủ:

```bash
python Do_an_XLA.py --input .\qrcodes\detection --no-display
```

Chạy có giao diện xem ảnh và vùng mã được phát hiện:

```bash
python Do_an_XLA.py --input .\Damaged
```

Tuỳ chọn hữu ích:

- `--show_steps`: hiển thị các bước tiền xử lý.
- `--pause`: dừng sau mỗi ảnh khi đang hiển thị.
- `--strong-only`: chỉ dùng WeChat, ZXing và OpenCV để giảm false positive.
- `--download_models`: tải lại các model còn thiếu.
- `--model_dir PATH`: chỉ định thư mục model khác.

## Cấu trúc chính

```text
Do_an_XLA.py       # mã nguồn duy nhất
wechat_models/     # model WeChatQR
qrcodes/           # dữ liệu thử nghiệm
Damaged/           # ảnh đầu vào bổ sung
```

Kết quả/debug được tạo trong `out_qr/` nếu có công cụ khác sinh ra thư mục này; thư mục đó không được đưa vào Git.
