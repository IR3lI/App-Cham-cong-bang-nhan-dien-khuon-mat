"""
Face Detector Module — RetinaFace (InsightFace) + Tiền xử lý ảnh
=================================================================
Phát hiện khuôn mặt bằng RetinaFace (primary) hoặc MTCNN (fallback).
Tính năng nâng cấp:
  - 5-point landmark face alignment (chuẩn ArcFace 112×112)
  - CLAHE equalization — cải thiện mặt trong điều kiện sáng yếu
  - Gaussian denoising nhẹ
  - Blur detection (bỏ frame bị rung/mờ)
  - Frame auto-downscale trước detect (giảm tải CPU/GPU)
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional
import warnings

# ─── Backend flags ────────────────────────────────────────────────────────────
_INSIGHTFACE_AVAILABLE = False
_MTCNN_AVAILABLE = False

try:
    import insightface
    from insightface.app import FaceAnalysis
    _INSIGHTFACE_AVAILABLE = True
except ImportError:
    pass

if not _INSIGHTFACE_AVAILABLE:
    try:
        from facenet_pytorch import MTCNN
        import torch
        _MTCNN_AVAILABLE = True
    except ImportError:
        pass

if not _INSIGHTFACE_AVAILABLE and not _MTCNN_AVAILABLE:
    warnings.warn(
        "[FaceDetector] Không tìm thấy insightface lẫn facenet-pytorch!\n"
        "  Cài insightface: pip install insightface onnxruntime-gpu\n"
        "  Hoặc fallback:   pip install facenet-pytorch"
    )


# ─── Cấu hình ────────────────────────────────────────────────────────────────
FACE_OUTPUT_SIZE  = 112       # ArcFace standard input size (pixels)
MIN_FACE_SIZE     = 30        # Bỏ qua mặt nhỏ hơn N pixel
DETECT_CONF_THRES = 0.50      # Ngưỡng confidence RetinaFace
BLUR_VAR_THRESHOLD = 30.0     # Laplacian variance — frame < này bị bỏ qua
                               # (Giảm từ 80 → 30: webcam thường có variance thấp hơn camera chất lượng cao)
MAX_FRAME_LONG_SIDE = 1280    # Downscale frame nếu lớn hơn (px)

# ArcFace reference landmarks — 5-point chuẩn 112×112
ARCFACE_REF_LANDMARKS = np.array([
    [38.2946, 51.6963],   # eye_left
    [73.5318, 51.5014],   # eye_right
    [56.0252, 71.7366],   # nose
    [41.5493, 92.3655],   # mouth_left
    [70.7299, 92.2041],   # mouth_right
], dtype=np.float32)

# CLAHE parameters
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


# ─── Helper functions ─────────────────────────────────────────────────────────

def _estimate_norm(lmk: np.ndarray, image_size: int = 112) -> np.ndarray:
    """
    Tính ma trận affine 2×3 để align 5 landmarks → ARCFACE_REF_LANDMARKS.
    Sử dụng cv2.estimateAffinePartial2D (similarity transform: scale + rotate + translate).
    """
    assert lmk.shape == (5, 2), f"Cần 5 landmarks, got {lmk.shape}"
    ref = ARCFACE_REF_LANDMARKS.copy()
    # Scale ref về image_size (mặc định ref đã ở 112×112)
    if image_size != 112:
        scale = image_size / 112.0
        ref *= scale
    M, _ = cv2.estimateAffinePartial2D(lmk, ref, method=cv2.RANSAC, ransacReprojThreshold=5)
    return M


def _align_face(img_bgr: np.ndarray, landmarks: np.ndarray,
                size: int = FACE_OUTPUT_SIZE) -> np.ndarray:
    """
    Warp ảnh theo ma trận affine → khuôn mặt chuẩn ArcFace.

    Args:
        img_bgr:   Frame gốc BGR (full frame).
        landmarks: numpy (5, 2) — tọa độ landmarks trên frame gốc.
        size:      Kích thước output (mặc định 112).

    Returns:
        numpy (size, size, 3) BGR — mặt đã align.
    """
    M = _estimate_norm(landmarks.astype(np.float32), size)
    if M is None:
        # Fallback: crop center và resize nếu không tính được affine
        h, w = img_bgr.shape[:2]
        crop = img_bgr[h//4:3*h//4, w//4:3*w//4]
        return cv2.resize(crop, (size, size))
    warped = cv2.warpAffine(img_bgr, M, (size, size), flags=cv2.INTER_LINEAR)
    return warped


def preprocess_face(face_bgr: np.ndarray) -> np.ndarray:
    """
    Pipeline tiền xử lý khuôn mặt đã crop/align (112×112 BGR):

    1. CLAHE trên kênh L (LAB space) — cải thiện độ tương phản cục bộ
    2. Gaussian blur nhẹ — giảm noise camera
    3. Clip và chuyển về uint8

    Args:
        face_bgr: numpy (H, W, 3) BGR.

    Returns:
        numpy (H, W, 3) BGR đã tiền xử lý.
    """
    # ── 1. CLAHE trên kênh L (LAB) ───────────────────────────
    lab = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    l_eq = _CLAHE.apply(l_ch)
    lab_eq = cv2.merge([l_eq, a_ch, b_ch])
    result = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    # ── 2. Gaussian denoise nhẹ (kernel 3×3, sigma 0.5) ──────
    result = cv2.GaussianBlur(result, (3, 3), sigmaX=0.5)

    return result


def preprocess_frame(frame_bgr: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Tiền xử lý frame toàn cảnh trước khi đưa vào detector:
    - Downscale nếu cạnh dài > MAX_FRAME_LONG_SIDE
    - Trả về scale factor để map coordinates về frame gốc

    Args:
        frame_bgr: Frame BGR gốc.

    Returns:
        (frame_scaled, scale): frame sau downscale và tỉ lệ scale.
    """
    h, w = frame_bgr.shape[:2]
    long_side = max(h, w)
    if long_side <= MAX_FRAME_LONG_SIDE:
        return frame_bgr, 1.0

    scale = MAX_FRAME_LONG_SIDE / long_side
    new_w = int(w * scale)
    new_h = int(h * scale)
    frame_scaled = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return frame_scaled, scale


def is_blurry(frame_bgr: np.ndarray, threshold: float = BLUR_VAR_THRESHOLD) -> bool:
    """
    Kiểm tra frame có bị mờ/rung quá không bằng Laplacian variance.
    Frame mờ → variance nhỏ → bỏ qua để tiết kiệm compute.

    Returns:
        True nếu frame BỊ MỜ (nên bỏ qua).
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return lap_var < threshold


# ─── InsightFace Backend ─────────────────────────────────────────────────────

class _InsightFaceBackend:
    """
    Backend dùng insightface.app.FaceAnalysis (Buffalo_s model — nhẹ, nhanh).
    Tự động dùng GPU nếu CUDA khả dụng, fallback về CPU.
    """

    def __init__(self, device: str = None):
        import torch
        ctx_id = 0 if (device == "cuda" or (device is None and torch.cuda.is_available())) else -1

        # Buffalo_s: RetinaFace detector + 2D106 landmarks (nhẹ)
        # Buffalo_l: chính xác hơn nhưng nặng hơn
        self.app = FaceAnalysis(
            name="buffalo_sc",       # buffalo_sc = RetinaFace + gaze (nhẹ nhất)
            allowed_modules=["detection"],   # Chỉ detect, không recognition
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
                       if ctx_id == 0
                       else ["CPUExecutionProvider"],
        )
        self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))
        dev_name = "CUDA" if ctx_id == 0 else "CPU"
        print(f"[FaceDetector] InsightFace (RetinaFace) initialized on {dev_name}")

    def detect(self, frame_bgr: np.ndarray):
        """
        Returns list of dicts: {bbox, kps, det_score}
        bbox: [x1,y1,x2,y2], kps: (5,2) landmarks
        """
        # InsightFace cần BGR (giống OpenCV) — không cần convert
        faces = self.app.get(frame_bgr)
        return faces


# ─── MTCNN Backend (fallback) ────────────────────────────────────────────────

class _MTCNNBackend:
    """Fallback backend dùng MTCNN từ facenet-pytorch."""

    def __init__(self, device: str = None):
        import torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.mtcnn = MTCNN(
            image_size=FACE_OUTPUT_SIZE,
            margin=10,
            min_face_size=MIN_FACE_SIZE,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=False,
            device=self.device,
            keep_all=True,
        )
        print(f"[FaceDetector] MTCNN (fallback) initialized on {self.device}")

    def detect(self, frame_bgr: np.ndarray):
        """Returns list of dicts: {bbox, kps, det_score}"""
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        boxes, probs, points = self.mtcnn.detect(frame_rgb, landmarks=True)
        if boxes is None:
            return []

        results = []
        for box, prob, pts in zip(boxes, probs, points):
            if prob is None or prob < DETECT_CONF_THRES:
                continue
            results.append({
                "bbox":      np.array(box, dtype=np.float32),
                "kps":       pts.astype(np.float32),   # (5,2)
                "det_score": float(prob),
            })
        return results


# ─── FaceDetector (public API) ───────────────────────────────────────────────

class FaceDetector:
    """
    Wrapper phát hiện và align khuôn mặt từ frame OpenCV (BGR).

    Ưu tiên InsightFace (RetinaFace); tự động fallback về MTCNN.

    Sử dụng:
        detector = FaceDetector()
        faces, boxes, landmarks = detector.detect(frame_bgr)
    """

    def __init__(self, device: str = None, min_face_size: int = MIN_FACE_SIZE,
                 apply_preprocessing: bool = True):
        """
        Args:
            device:              "cuda" | "cpu" | None (tự phát hiện)
            min_face_size:       Bỏ qua mặt nhỏ hơn N pixel
            apply_preprocessing: Có chạy CLAHE + denoise sau khi align không
        """
        self.min_face_size    = min_face_size
        self.apply_preprocess = apply_preprocessing
        self._backend_name    = "unknown"

        self._backend = self._init_backend(device)

    # ── Private ───────────────────────────────────────────────────────────────

    def _init_backend(self, device: str):
        if _INSIGHTFACE_AVAILABLE:
            try:
                b = _InsightFaceBackend(device=device)
                self._backend_name = "insightface"
                return b
            except Exception as e:
                print(f"[FaceDetector] InsightFace init thất bại: {e}")
                print("[FaceDetector] Thử fallback MTCNN...")

        if _MTCNN_AVAILABLE:
            b = _MTCNNBackend(device=device)
            self._backend_name = "mtcnn"
            return b

        raise ImportError(
            "Không tìm thấy backend phát hiện mặt!\n"
            "  Cài insightface: pip install insightface onnxruntime-gpu\n"
            "  Hoặc MTCNN:      pip install facenet-pytorch"
        )

    def _process_face(self, frame_bgr: np.ndarray,
                      face_dict: dict,
                      frame_scale: float) -> Tuple[np.ndarray, List[int], Optional[np.ndarray]]:
        """
        Từ 1 kết quả detect → (face_crop, box_original, landmarks_original).

        Args:
            frame_bgr:   Frame GỐC (trước downscale) để crop chất lượng cao.
            face_dict:   Dict {bbox, kps, det_score} từ backend.
            frame_scale: Tỉ lệ downscale đã áp dụng lúc detect (để map coords).

        Returns:
            face_aligned: (112,112,3) BGR sau align + preprocess.
            box_orig:     [x1,y1,x2,y2] trên frame gốc.
            kps_orig:     (5,2) landmarks trên frame gốc.
        """
        h, w = frame_bgr.shape[:2]

        # Map tọa độ về frame gốc (nếu đã downscale)
        inv_scale = 1.0 / frame_scale
        bbox = face_dict["bbox"] * inv_scale
        kps  = face_dict.get("kps")

        x1, y1, x2, y2 = [int(c) for c in bbox]
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(w, x2); y2 = min(h, y2)

        if (x2 - x1) < self.min_face_size or (y2 - y1) < self.min_face_size:
            return None, None, None

        # Scale landmarks về frame gốc
        kps_orig = None
        if kps is not None:
            kps_orig = kps * inv_scale

        # ── Face alignment ────────────────────────────────────
        if kps_orig is not None:
            face_aligned = _align_face(frame_bgr, kps_orig, size=FACE_OUTPUT_SIZE)
        else:
            # Không có landmarks → crop thô
            crop = frame_bgr[y1:y2, x1:x2]
            face_aligned = cv2.resize(crop, (FACE_OUTPUT_SIZE, FACE_OUTPUT_SIZE))

        # ── Tiền xử lý (CLAHE + denoise) ─────────────────────
        if self.apply_preprocess:
            face_aligned = preprocess_face(face_aligned)

        box_orig = [x1, y1, x2, y2]
        return face_aligned, box_orig, kps_orig

    # ── Public API ────────────────────────────────────────────────────────────

    def detect(
        self, frame_bgr: np.ndarray, skip_blur_check: bool = False
    ) -> Tuple[List[np.ndarray], List[List[int]], List[Optional[np.ndarray]]]:
        """
        Phát hiện, align và tiền xử lý tất cả khuôn mặt trong frame.

        Args:
            frame_bgr:        numpy HxWx3 BGR (từ cv2.VideoCapture).
            skip_blur_check:  Bỏ qua kiểm tra blur (dùng khi đăng ký).

        Returns:
            faces_bgr:  list (112,112,3) BGR — mặt đã align + preprocess.
            boxes:      list [x1,y1,x2,y2] — bounding boxes trên frame gốc.
            landmarks:  list (5,2) hoặc None — facial landmarks trên frame gốc.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return [], [], []

        # ── Blur check (bỏ qua frame rung/mờ) ────────────────
        if not skip_blur_check and is_blurry(frame_bgr):
            return [], [], []

        # ── Frame downscale (giảm tải compute) ───────────────
        frame_scaled, scale = preprocess_frame(frame_bgr)

        # ── Detect ───────────────────────────────────────────
        try:
            raw_faces = self._backend.detect(frame_scaled)
        except Exception as e:
            print(f"[FaceDetector] Lỗi detect: {e}")
            return [], [], []

        if not raw_faces:
            return [], [], []

        # ── Filter confidence ─────────────────────────────────
        raw_faces = [f for f in raw_faces
                     if f.get("det_score", 1.0) >= DETECT_CONF_THRES]

        # ── Process từng mặt ─────────────────────────────────
        faces_bgr, boxes, lm_list = [], [], []
        for face_dict in raw_faces:
            face, box, kps = self._process_face(frame_bgr, face_dict, scale)
            if face is not None:
                faces_bgr.append(face)
                boxes.append(box)
                lm_list.append(kps)

        return faces_bgr, boxes, lm_list

    def detect_largest(
        self, frame_bgr: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[List[int]], Optional[np.ndarray]]:
        """
        Trả về khuôn mặt lớn nhất trong frame (dùng cho Registration Mode).

        Returns:
            face_bgr:  (112,112,3) BGR hoặc None.
            box:       [x1,y1,x2,y2] hoặc None.
            landmarks: (5,2) hoặc None.
        """
        faces, boxes, lm_list = self.detect(frame_bgr, skip_blur_check=True)
        if not faces:
            return None, None, None

        areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
        idx = int(np.argmax(areas))
        return faces[idx], boxes[idx], lm_list[idx]

    @property
    def backend_name(self) -> str:
        """Tên backend đang dùng: 'insightface' hoặc 'mtcnn'."""
        return self._backend_name


# ── Singleton ─────────────────────────────────────────────────────────────────
_detector_instance: Optional[FaceDetector] = None


def get_detector() -> FaceDetector:
    """Trả về singleton FaceDetector (lazy init)."""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = FaceDetector()
    return _detector_instance
