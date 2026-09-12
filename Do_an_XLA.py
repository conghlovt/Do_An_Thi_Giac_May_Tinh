import os
import time
import argparse
from pathlib import Path

import cv2
import numpy as np
import requests
import zxingcpp

# Thư viện tuỳ chọn (nếu có thì dùng thêm):
try:
    from pyzbar.pyzbar import decode as decode_pyzbar
except Exception:
    decode_pyzbar = None

try:
    from pylibdmtx.pylibdmtx import decode as decode_dmtx
except Exception:
    decode_dmtx = None


# =========================
# Cấu hình
# =========================
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = BASE_DIR / "wechat_models"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

ENGINE_PRIORITY = {
    "WeChat": 5,
    "ZXing": 4,
    "OpenCVQR": 3,
    "ZBar": 2,
    "DataMatrix": 2,
}

# Bỏ kết quả text quá ngắn (thường là rác).
MIN_TEXT_LEN = 4

# Lọc text rác: nếu tỷ lệ ký tự điều khiển cao thì loại.
MAX_CONTROL_CHAR_RATIO = 0.12  # Tối đa 12% ký tự điều khiển

# Lọc hình học cho tứ giác (kích thước/tỷ lệ/độ lồi...).
MIN_AREA_FRAC = 0.0007  # Tối thiểu ~0.07% diện tích ảnh
MAX_AREA_FRAC = 0.55  # Tối đa 55% diện tích ảnh
MIN_SIDE_PX = 22  # Cạnh tối thiểu (px)
MAX_SIDE_RATIO = 3.2  # max(cạnh)/min(cạnh) quá lớn ⇒ loại (hình quá dài)
BBOX_AR_MIN = 0.28  # Tỷ lệ bbox tối thiểu (w/h)
BBOX_AR_MAX = 3.6  # Tỷ lệ bbox tối đa (w/h)

# Loại trùng theo IoU (nhiều engine trả về cùng một mã).
DEDUP_IOU_THR = 0.55

# Nếu muốn “cứng” hơn (ít false positives hơn), có thể bật tuỳ chọn dưới đây:
# ONLY_STRONG_ENGINES = True  # Chỉ dùng WeChat/ZXing/OpenCVQR (giảm FP nhưng có thể bỏ sót).
ONLY_STRONG_ENGINES = False
DECODE_SCALES = (1.0, 1.6, 2.2)
DECODE_ROTATION_GROUPS = ((0, 90, 180, 270), (45, 135, 225, 315))


# =========================
# Tải model WeChatQR (tự động nếu thiếu)
# =========================
def download_wechat_models(model_dir=DEFAULT_MODEL_DIR):
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    urls = {
        "detect.prototxt": "https://raw.githubusercontent.com/WeChatCV/opencv_3rdparty/wechat_qrcode/detect.prototxt",
        "detect.caffemodel": "https://raw.githubusercontent.com/WeChatCV/opencv_3rdparty/wechat_qrcode/detect.caffemodel",
        "sr.prototxt": "https://raw.githubusercontent.com/WeChatCV/opencv_3rdparty/wechat_qrcode/sr.prototxt",
        "sr.caffemodel": "https://raw.githubusercontent.com/WeChatCV/opencv_3rdparty/wechat_qrcode/sr.caffemodel",
    }

    print("⏳ Checking & downloading WeChat QR models...")
    for name, url in urls.items():
        path = model_dir / name
        if not path.exists():
            try:
                r = requests.get(url, timeout=30)
                r.raise_for_status()
                with open(path, "wb") as f:
                    f.write(r.content)
                print(f"✅ Downloaded: {name}")
            except Exception as e:
                print(f"⚠️ Failed to download {name}: {e}")
    print("✅ WeChat models ready.")


def wechat_models_ok(model_dir=DEFAULT_MODEL_DIR) -> bool:
    model_dir = Path(model_dir)
    need = ["detect.prototxt", "detect.caffemodel", "sr.prototxt", "sr.caffemodel"]
    return all((model_dir / n).is_file() for n in need)


# =========================
# Hàm hỗ trợ hiển thị/debug
# =========================
def _to_bgr(img):
    if img is None:
        return None
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def _resize_max_side(img, max_side=900):
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return img
    scale = max_side / m
    return cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)


def add_caption_below(img_bgr, caption, cap_h=34):
    h, w = img_bgr.shape[:2]
    strip = np.zeros((cap_h, w, 3), dtype=img_bgr.dtype)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.65
    thickness = 2

    (tw, th), _ = cv2.getTextSize(caption, font, font_scale, thickness)
    while tw > w - 20 and font_scale > 0.35:
        font_scale -= 0.05
        (tw, th), _ = cv2.getTextSize(caption, font, font_scale, thickness)

    x = max(10, (w - tw) // 2)
    y = (cap_h + th) // 2
    cv2.putText(strip, caption, (x, y), font, font_scale, (0, 255, 255), thickness, cv2.LINE_AA)
    return np.vstack([img_bgr, strip])


def show_window(win_name, img, wait_ms=1, max_side=1000):
    if img is None:
        return
    vis = _to_bgr(img)
    vis = _resize_max_side(vis, max_side=max_side)
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.imshow(win_name, vis)
    cv2.waitKey(wait_ms)


def show_montage(win_name, images, labels, ncols=3, cap_h=34, tile_max_side=380, wait_ms=1):
    tiles = []
    for img, lab in zip(images, labels):
        if img is None:
            continue
        t = _to_bgr(img)
        t = _resize_max_side(t, max_side=tile_max_side)
        t = add_caption_below(t, lab, cap_h=cap_h)
        tiles.append(t)
    if not tiles:
        return

    rows = []
    for i in range(0, len(tiles), ncols):
        row = tiles[i:i + ncols]
        max_h = max(im.shape[0] for im in row)
        padded = []
        for im in row:
            if im.shape[0] < max_h:
                pad = np.zeros((max_h - im.shape[0], im.shape[1], 3), dtype=im.dtype)
                im = np.vstack([im, pad])
            padded.append(im)
        rows.append(np.hstack(padded))

    max_w = max(r.shape[1] for r in rows)
    padded_rows = []
    for r in rows:
        if r.shape[1] < max_w:
            pad = np.zeros((r.shape[0], max_w - r.shape[1], 3), dtype=r.dtype)
            r = np.hstack([r, pad])
        padded_rows.append(r)

    final = np.vstack(padded_rows)
    show_window(win_name, final, wait_ms=wait_ms, max_side=1400)


# =========================
# Hàm hỗ trợ xử lý ảnh & hình học (tăng tỷ lệ decode)
# =========================
def order_points(pts: np.ndarray) -> np.ndarray:
    pts = np.array(pts, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] < 4:
        return None
    if pts.shape[0] > 4:
        rect = cv2.minAreaRect(pts.astype(np.float32))
        pts = cv2.boxPoints(rect).astype(np.float32)

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def bbox_from_pts(pts):
    xs = pts[:, 0]
    ys = pts[:, 1]
    x1, y1 = float(xs.min()), float(ys.min())
    x2, y2 = float(xs.max()), float(ys.max())
    return (x1, y1, x2, y2)


def bbox_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    union = area_a + area_b - inter + 1e-9
    return inter / union


def normalize_quad(pts, w, h):
    if pts is None:
        return None
    pts = np.array(pts, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] < 4:
        return None
    if pts.shape[0] != 4:
        rect = cv2.minAreaRect(pts.astype(np.float32))
        pts = cv2.boxPoints(rect).astype(np.float32)

    pts = order_points(pts)
    if pts is None:
        return None

    # Clip điểm về trong biên ảnh để vẽ polygon không bị “văng” ra ngoài.
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
    return pts


def quad_ok(pts, w, h):
    pts = pts.astype(np.float32)
    area = float(cv2.contourArea(pts))
    img_area = float(w * h)

    if area < max(500.0, MIN_AREA_FRAC * img_area):
        return False
    if area > MAX_AREA_FRAC * img_area:
        return False

    (tl, tr, br, bl) = pts
    sides = [
        np.linalg.norm(tr - tl),
        np.linalg.norm(br - tr),
        np.linalg.norm(bl - br),
        np.linalg.norm(tl - bl),
    ]
    if min(sides) < MIN_SIDE_PX:
        return False
    if (max(sides) / (min(sides) + 1e-9)) > MAX_SIDE_RATIO:
        return False

    x1, y1, x2, y2 = bbox_from_pts(pts)
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    ar = bw / bh
    if ar < BBOX_AR_MIN or ar > BBOX_AR_MAX:
        return False

    # Kiểm tra tứ giác có lồi (convex) hay không.
    if not cv2.isContourConvex(pts.astype(np.int32)):
        return False

    return True


def text_ok(s: str) -> bool:
    if s is None:
        return False
    s = s.strip()
    if len(s) < MIN_TEXT_LEN:
        return False

    # Tính tỷ lệ ký tự điều khiển trong text (để loại rác).
    ctrl = 0
    for ch in s:
        o = ord(ch)
        if o < 32 and ch not in ("\n", "\r", "\t"):
            ctrl += 1
    if len(s) > 0 and (ctrl / len(s)) > MAX_CONTROL_CHAR_RATIO:
        return False

    return True


def filter_and_dedup_results(results, w, h, iou_thr=DEDUP_IOU_THR):
    cleaned = []
    for r in results:
        txt = (r.get("text") or "").strip()
        if not text_ok(txt):
            continue

        pts = normalize_quad(r.get("points"), w, h)
        if pts is None:
            continue
        if not quad_ok(pts, w, h):
            continue

        bb = bbox_from_pts(pts)
        rr = dict(r)
        rr["text"] = txt
        rr["points"] = pts
        rr["bbox"] = bb
        cleaned.append(rr)

    # Sắp xếp theo: độ ưu tiên engine → độ dài text → diện tích vùng phát hiện.
    def score(rr):
        pri = ENGINE_PRIORITY.get(rr.get("engine", ""), 0)
        ln = len(rr.get("text", ""))
        area = cv2.contourArea(rr["points"].astype(np.float32))
        return pri * 100000 + ln * 100 + area

    cleaned.sort(key=score, reverse=True)

    kept = []
    for r in cleaned:
        dup = False
        for k in kept:
            if bbox_iou(r["bbox"], k["bbox"]) >= iou_thr:
                dup = True
                break
        if not dup:
            kept.append(r)

    return kept


def warp_by_points(img, points):
    result = warp_by_points_with_matrix(img, points)
    return result[0] if result is not None else None


def warp_by_points_with_matrix(img, points):
    """Warp a detected quadrilateral and return (image, source-to-ROI matrix)."""
    if points is None:
        return None

    h, w = img.shape[:2]
    pts = normalize_quad(points, w, h)
    if pts is None or len(pts) != 4:
        return None

    tl, tr, br, bl = pts
    max_width = max(2, int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl))))
    max_height = max(2, int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl))))
    dst = np.array(
        [[0, 0], [max_width - 1, 0],
         [max_width - 1, max_height - 1], [0, max_height - 1]],
        dtype=np.float32,
    )
    try:
        M = cv2.getPerspectiveTransform(pts, dst)
        warped = cv2.warpPerspective(img, M, (max_width, max_height))
        return warped, M
    except Exception:
        return None


def gamma_correct(gray, gamma=1.2):
    inv = 1.0 / max(gamma, 1e-6)
    table = (np.arange(256) / 255.0) ** inv * 255.0
    table = np.clip(table, 0, 255).astype(np.uint8)
    return cv2.LUT(gray, table)


def unsharp_mask(gray, amount=1.2, radius=2):
    blur = cv2.GaussianBlur(gray, (0, 0), radius)
    sharp = cv2.addWeighted(gray, 1.0 + amount, blur, -amount, 0)
    return sharp


def glare_inpaint(gray, thr=245):
    mask = (gray >= thr).astype(np.uint8) * 255
    if mask.mean() < 1:
        return gray
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=2)
    return cv2.inpaint(gray, mask, 3, cv2.INPAINT_TELEA)


# =========================
# Thu thập danh sách ảnh đầu vào (quét đệ quy)
# =========================
def collect_images(path: str):
    source = Path(path).expanduser()
    if source.is_file():
        return [str(source.resolve())]
    if not source.is_dir():
        return []

    return sorted(
        str(file.resolve())
        for file in source.rglob("*")
        if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
    )


# =========================
# PipelineMaster: điều phối toàn bộ pipeline
# =========================
class PipelineMaster:
    def __init__(
        self,
        show_steps=True,
        pause_each_image=False,
        model_dir=DEFAULT_MODEL_DIR,
        display=True,
        strong_only=ONLY_STRONG_ENGINES,
    ):
        self.model_dir = Path(model_dir)
        self.show_steps = show_steps
        self.pause_each_image = pause_each_image
        self.display = display
        self.strong_only = strong_only

        self.cv_qr = cv2.QRCodeDetector()

        self.detector = None
        try:
            from cv2 import wechat_qrcode
            self.detector = wechat_qrcode.WeChatQRCode(
                str(self.model_dir / "detect.prototxt"),
                str(self.model_dir / "detect.caffemodel"),
                str(self.model_dir / "sr.prototxt"),
                str(self.model_dir / "sr.caffemodel"),
            )
        except Exception as e:
            print("⚠️ WeChatQRCode init failed:", e)
            self.detector = None

    def pipeline_preprocess(self, img_bgr):
        stages = {}

        if img_bgr.ndim == 3:
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        else:
            gray = img_bgr.copy()
        stages["gray"] = gray

        # Khử vùng chói (glare) bằng inpaint.
        deglare = glare_inpaint(gray, thr=245)
        stages["deglare"] = deglare

        # Tăng tương phản cục bộ bằng CLAHE.
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(deglare)
        stages["enhanced"] = enhanced

        # Làm nét nhẹ (unsharp mask).
        sharp = unsharp_mask(enhanced, amount=1.0, radius=2)
        stages["sharp"] = sharp

        # Thử gamma correction ở 2 mức để bắt QR tối/tối hơn.
        stages["gamma_0.8"] = gamma_correct(sharp, gamma=0.8)
        stages["gamma_1.3"] = gamma_correct(sharp, gamma=1.3)

        # Giảm nhiễu nhẹ trước khi nhị phân hoá.
        gaussian = cv2.GaussianBlur(sharp, (5, 5), 0)
        denoised = cv2.medianBlur(gaussian, 3)
        stages["denoised"] = denoised

        # Nhị phân hoá (Adaptive/Otsu) + thêm bản đảo màu (invert).
        binary_adapt = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 21, 5
        )
        stages["binary_adapt"] = binary_adapt
        stages["binary_adapt_inv"] = cv2.bitwise_not(binary_adapt)

        _, binary_otsu = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        stages["binary_otsu"] = binary_otsu
        stages["binary_otsu_inv"] = cv2.bitwise_not(binary_otsu)

        # Giảm ảnh hưởng vùng bóng (ước lượng nền rồi chuẩn hoá).
        dilated = cv2.dilate(gray, np.ones((7, 7), np.uint8))
        bg_blur = cv2.medianBlur(dilated, 21)
        diff = 255 - cv2.absdiff(gray, bg_blur)
        norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
        stages["de_shadow"] = norm.astype(np.uint8)

        return stages

    def _decode_all_engines(self, img_gray):
        decoded = []
        points_only = []

        bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

        # --- WeChatQR (đa mã) ---
        if self.detector is not None:
            try:
                res, points = self.detector.detectAndDecode(bgr)
                if points is not None and len(points) > 0:
                    for i, pts in enumerate(points):
                        txt = ""
                        if res is not None and i < len(res):
                            txt = (res[i] or "").strip()
                        if txt:
                            decoded.append({"text": txt, "points": np.array(pts, dtype=np.float32), "engine": "WeChat"})
                        else:
                            points_only.append(np.array(pts, dtype=np.float32))
            except Exception:
                pass

        # --- OpenCV QRCodeDetector (đa mã) ---
        try:
            out = self.cv_qr.detectAndDecodeMulti(bgr)

            decoded_info, points = None, None
            if isinstance(out, tuple):
                if len(out) >= 4:
                    decoded_info, points = out[1], out[2]
                elif len(out) >= 3:
                    decoded_info, points = out[0], out[1]

            if points is not None and len(points) > 0:
                for i in range(len(points)):
                    pts = np.array(points[i], dtype=np.float32).reshape(-1, 2)
                    txt = ""
                    if decoded_info is not None and i < len(decoded_info):
                        txt = (decoded_info[i] or "").strip()
                    if txt:
                        decoded.append({"text": txt, "points": pts, "engine": "OpenCVQR"})
                    else:
                        points_only.append(pts)
        except Exception:
            pass

        # --- ZXing (đa mã) ---
        try:
            results = zxingcpp.read_barcodes(img_gray)
            for r in results:
                txt = (r.text or "").strip()
                if not txt:
                    continue
                pos = r.position
                pts = np.array([[p.x, p.y] for p in
                                [pos.top_left, pos.top_right, pos.bottom_right, pos.bottom_left]], dtype=np.float32)
                decoded.append({"text": txt, "points": pts, "engine": "ZXing"})
        except Exception:
            pass

        # Engine phụ (có thể sinh false positives) — có thể tắt bằng ONLY_STRONG_ENGINES.
        if not self.strong_only:
            # --- ZBar (tuỳ chọn) ---
            if decode_pyzbar is not None:
                try:
                    zs = decode_pyzbar(img_gray)
                    for z in zs:
                        txt = z.data.decode("utf-8", errors="ignore").strip()
                        if not txt:
                            continue

                        pts = None
                        if hasattr(z, "polygon") and z.polygon:
                            poly = np.array([[p.x, p.y] for p in z.polygon], dtype=np.float32)
                            if poly.shape[0] >= 4:
                                # Dùng minAreaRect để quy về 4 góc ổn định (tránh polygon méo/đường chéo dài).
                                rect = cv2.minAreaRect(poly.astype(np.float32))
                                pts = cv2.boxPoints(rect).astype(np.float32)

                        if pts is None:
                            rect = z.rect
                            pts = np.array([
                                [rect.left, rect.top],
                                [rect.left + rect.width, rect.top],
                                [rect.left + rect.width, rect.top + rect.height],
                                [rect.left, rect.top + rect.height],
                            ], dtype=np.float32)

                        decoded.append({"text": txt, "points": pts, "engine": "ZBar"})
                except Exception:
                    pass

            # --- DataMatrix (tuỳ chọn) ---
            if decode_dmtx is not None:
                try:
                    dms = decode_dmtx(img_gray, timeout=40)
                    for obj in dms:
                        txt = obj.data.decode("utf-8", errors="ignore").strip()
                        if not txt:
                            continue
                        rect = obj.rect
                        pts = np.array([
                            [rect.left, rect.top],
                            [rect.left + rect.width, rect.top],
                            [rect.left + rect.width, rect.top + rect.height],
                            [rect.left, rect.top + rect.height],
                        ], dtype=np.float32)
                        decoded.append({"text": txt, "points": pts, "engine": "DataMatrix"})
                except Exception:
                    pass

        # Chưa dedup ở đây vì cần lọc hình học trước rồi mới gộp kết quả.
        return decoded, points_only

    def decode_with_adjustment_loop(self, img_bgr, fname=""):
        # Chuẩn hoá kích thước ảnh đầu vào (tránh ảnh quá lớn gây chậm).
        H, W = img_bgr.shape[:2]
        base_scale = 1.0
        if max(H, W) > 1600:
            base_scale = 1600 / max(H, W)
        base_bgr = cv2.resize(img_bgr, None, fx=base_scale, fy=base_scale, interpolation=cv2.INTER_AREA)

        stages = self.pipeline_preprocess(base_bgr)

        if self.display and self.show_steps:
            show_montage(
                "PREPROCESS",
                images=[stages["gray"], stages["deglare"], stages["enhanced"],
                        stages["sharp"], stages["binary_adapt"], stages["binary_otsu"]],
                labels=[f"{fname} | 1-Gray", "1b-DeGlare", "2-CLAHE",
                        "2b-Sharpen", "4-Binary(Adapt)", "4-Binary(Otsu)"],
                ncols=3,
                wait_ms=1
            )

        candidates = [
            ("Sharp", stages["sharp"]),
            ("Gamma0.8", stages["gamma_0.8"]),
            ("Gamma1.3", stages["gamma_1.3"]),
            ("DeShadow", stages["de_shadow"]),
            ("BinaryAdapt", stages["binary_adapt"]),
            ("BinaryAdaptInv", stages["binary_adapt_inv"]),
            ("BinaryOtsu", stages["binary_otsu"]),
            ("BinaryOtsuInv", stages["binary_otsu_inv"]),
        ]

        # Resize each candidate once per scale. The previous implementation
        # repeated the same resize for every rotation, adding unnecessary work.
        scaled_candidates = {
            scale_val: [
                (
                    name,
                    image
                    if scale_val == 1.0
                    else cv2.resize(
                        image,
                        None,
                        fx=scale_val,
                        fy=scale_val,
                        interpolation=cv2.INTER_CUBIC,
                    ),
                )
                for name, image in candidates
            ]
            for scale_val in DECODE_SCALES
        }

        best_points_only = None
        best_work = None

        for rot_list in DECODE_ROTATION_GROUPS:
            for scale_val in DECODE_SCALES:
                for angle in rot_list:
                    for name, img_gray in scaled_candidates[scale_val]:
                        work = img_gray

                        if angle != 0:
                            rows, cols = work.shape[:2]
                            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1.0)
                            work = cv2.warpAffine(work, M, (cols, rows))

                        decoded_raw, pts_only = self._decode_all_engines(work)

                        # ✅ Lọc hình học + loại trùng ngay trong lần thử này.
                        decoded = filter_and_dedup_results(decoded_raw, work.shape[1], work.shape[0])

                        if decoded:
                            info = f"{name} | rot={angle} | s={scale_val} | kept={len(decoded)}"
                            return decoded, work, info

                        if pts_only and best_points_only is None:
                            best_points_only = pts_only
                            best_work = work

        # Nếu chỉ detect được điểm mà chưa decode ra text: warp ROI rồi decode lại.
        if best_points_only is not None and best_work is not None:
            final_decoded = []
            for pts in best_points_only[:8]:
                warped = warp_by_points_with_matrix(best_work, pts)
                if warped is None:
                    continue
                roi, source_to_roi = warped
                if roi.shape[0] < 40 or roi.shape[1] < 40:
                    continue

                roi_scale = 2.2
                roi_up = cv2.resize(roi, None, fx=roi_scale, fy=roi_scale, interpolation=cv2.INTER_CUBIC)
                decoded_raw2, _ = self._decode_all_engines(roi_up)

                # Lọc kết quả trong không gian ROI.
                decoded2 = filter_and_dedup_results(decoded_raw2, roi_up.shape[1], roi_up.shape[0])
                try:
                    roi_to_source = np.linalg.inv(source_to_roi)
                    for result in decoded2:
                        roi_points = np.asarray(result["points"], dtype=np.float32).reshape(-1, 1, 2)
                        roi_points /= roi_scale
                        result["points"] = cv2.perspectiveTransform(roi_points, roi_to_source).reshape(-1, 2)
                        result["bbox"] = bbox_from_pts(result["points"])
                except (np.linalg.LinAlgError, cv2.error):
                    continue
                final_decoded.extend(decoded2)

            # Gộp theo text để tránh trùng khi các ROI khác nhau.
            uniq = {}
            for d in final_decoded:
                if d["text"] not in uniq:
                    uniq[d["text"]] = d
            final_decoded = list(uniq.values())

            if final_decoded:
                return final_decoded, best_work, f"WarpROI-Redecode | kept={len(final_decoded)}"

        return [], None, None

    def run(self, input_path):
        files = collect_images(input_path)
        if not files:
            print(f"❌ No images found in: {input_path}")
            return

        print(f"🚀 START: {len(files)} images | input={os.path.abspath(input_path)}")
        print("=" * 70)

        ok_images = 0
        t0 = time.time()

        for fpath in files:
            fname = os.path.basename(fpath)
            img = cv2.imread(fpath)
            if img is None:
                print(f"⚠️ Cannot read: {fpath}")
                continue

            if self.display:
                show_window("INPUT", add_caption_below(_resize_max_side(img, 1000), f"INPUT | {fname}"), wait_ms=1)

            decoded_list, work_img, info = self.decode_with_adjustment_loop(img, fname=fname)

            if decoded_list:
                ok_images += 1
                print(f"✅ {fname} | {info}")
                for i, d in enumerate(decoded_list, 1):
                    print(f"   [{i}] {d['engine']}: {d['text'][:150]}")

                # Vẽ polygon minh hoạ vùng QR đã nhận diện.
                if work_img is None:
                    work_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                vis = cv2.cvtColor(work_img, cv2.COLOR_GRAY2BGR) if work_img.ndim == 2 else work_img.copy()

                for i, d in enumerate(decoded_list, 1):
                    pts = d.get("points", None)
                    if pts is None:
                        continue
                    pts_ord = normalize_quad(pts, vis.shape[1], vis.shape[0])
                    if pts_ord is None:
                        continue
                    pts_i = pts_ord.astype(int).reshape((-1, 1, 2))
                    cv2.polylines(vis, [pts_i], True, (0, 255, 0), 2)

                    x, y = int(pts_i[0, 0, 0]), int(pts_i[0, 0, 1])
                    cv2.putText(vis, f"{i}", (x, max(0, y - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                if self.display:
                    show_window(
                        "RESULT",
                        add_caption_below(_resize_max_side(vis, 1400), f"RESULT | {fname} | {len(decoded_list)} code(s)"),
                        wait_ms=1,
                        max_side=1400,
                    )
            else:
                print(f"❌ {fname} | Not found")

            if self.pause_each_image and self.display:
                print("Press any key to continue...")
                cv2.waitKey(0)

        total = len(files)
        print("=" * 70)
        print(f"📊 RESULT: {ok_images}/{total} images have >=1 code ({(ok_images / total * 100):.1f}%)")
        elapsed = time.perf_counter() - t0
        print(f"⏱️ Time: {elapsed:.2f}s")
        if self.display:
            print("Close windows or press any key to exit.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return {"total": total, "success": ok_images, "elapsed_seconds": elapsed}


# =========================
# Chương trình chính
# =========================
def main():
    parser = argparse.ArgumentParser(description="Decode QR/Data Matrix codes from images.")
    parser.add_argument("--input", required=True, help="Folder path OR single image path")
    parser.add_argument("--show_steps", action="store_true", help="Show preprocess montage per image")
    parser.add_argument("--pause", action="store_true", help="Pause each image until key pressed")
    parser.add_argument("--no-display", action="store_true", help="Run without OpenCV windows")
    parser.add_argument("--strong-only", action="store_true", help="Use only WeChat, ZXing and OpenCV engines")
    parser.add_argument("--download_models", action="store_true", help="Force download WeChat models")
    parser.add_argument(
        "--model_dir",
        default=str(DEFAULT_MODEL_DIR),
        help="WeChat model folder (default: project/wechat_models)",
    )
    args = parser.parse_args()

    if args.download_models or (not wechat_models_ok(args.model_dir)):
        download_wechat_models(args.model_dir)

    pipeline = PipelineMaster(
        show_steps=args.show_steps,
        pause_each_image=args.pause,
        model_dir=args.model_dir,
        display=not args.no_display,
        strong_only=args.strong_only,
    )
    pipeline.run(args.input)


if __name__ == "__main__":
    main()
