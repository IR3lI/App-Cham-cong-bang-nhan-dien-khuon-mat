"""
Embedding Extractor Module
Tải model IResNet-50 từ file .pth và trích xuất ArcFace embedding vector 512-d.
Nhận input là khuôn mặt đã align (112×112) từ FaceDetector (RetinaFace).
"""

import os
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.iresnet import iresnet50

# ─── Cấu hình ────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "ir50_scface_best_arcface_unfreze_all.pth",
)
EMBEDDING_DIM = 512
IMG_SIZE = 112  # ArcFace standard input size


class EmbeddingExtractor:
    """
    Load model IResNet-50 (ArcFace) và trích xuất embedding vector từ ảnh khuôn mặt.

    Sử dụng:
        extractor = EmbeddingExtractor()
        embedding = extractor.get_embedding(face_image_np)  # numpy HxWx3 BGR
    """

    def __init__(self, model_path: str = MODEL_PATH, device: str = None):
        self.device = self._resolve_device(device)
        self.model = self._load_model(model_path)
        self.transform = self._build_transform()
        print(f"[EmbeddingExtractor] Model loaded on {self.device}")

    # ── Private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device:
            return torch.device(device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def _safe_load(model_path: str, map_loc="cpu"):
        """
        Load checkpoint with multiple fallback strategies:
        1. torch.load standard (zip-based, PyTorch >= 1.6)
        2. Legacy pickle format
        3. torch.load with encoding='latin1' (Python 2 checkpoint)
        """
        # Strategy 1: Standard torch.load (zip format)
        try:
            return torch.load(model_path, map_location=map_loc, weights_only=False)
        except RuntimeError as e:
            err_msg = str(e).lower()
            if "zip" in err_msg or "multidisk" in err_msg or "unsupported" in err_msg:
                print("[EmbeddingExtractor] ZIP load failed, trying legacy pickle...")
            else:
                raise

        # Strategy 2: Legacy pickle (PyTorch < 1.6)
        try:
            with open(model_path, "rb") as f:
                data = pickle.load(f)
            print("[EmbeddingExtractor] Loaded via legacy pickle format.")
            return data
        except Exception as e2:
            print(f"[EmbeddingExtractor] Pickle load also failed: {type(e2).__name__}: {e2}")

        # Strategy 3: torch.load with encoding='latin1' (Python 2 compat)
        try:
            return torch.load(model_path, map_location=map_loc,
                              weights_only=False, encoding="latin1")
        except Exception as e3:
            raise RuntimeError(
                f"Cannot load checkpoint from:\n  {model_path}\n"
                f"Last error: {e3}\n"
                "Check: (1) Is the .pth file corrupted? "
                "(2) Is the architecture IResNet-50?"
            ) from e3

    def _load_model(self, model_path: str) -> torch.nn.Module:
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Không tìm thấy file trọng số: {model_path}\n"
                "Đảm bảo file .pth nằm trong thư mục gốc dự án."
            )

        model = iresnet50(num_features=EMBEDDING_DIM)

        # Load checkpoint với fallback
        checkpoint = self._safe_load(model_path, map_loc=self.device)

        if isinstance(checkpoint, dict):
            # Thử các key phổ biến
            state_dict = (
                checkpoint.get("state_dict")
                or checkpoint.get("model")
                or checkpoint.get("model_state_dict")
                or checkpoint
            )
        else:
            state_dict = checkpoint

        # Xử lý prefix 'module.' nếu được train với DataParallel
        new_state = {}
        for k, v in state_dict.items():
            new_key = k.replace("module.", "")
            new_state[new_key] = v

        missing, unexpected = model.load_state_dict(new_state, strict=False)
        if missing:
            print(f"[EmbeddingExtractor] Keys thiếu ({len(missing)}): {missing[:3]}...")
        if unexpected:
            print(f"[EmbeddingExtractor] Keys thừa ({len(unexpected)}): {unexpected[:3]}...")

        model.eval()
        model.to(self.device)
        return model

    @staticmethod
    def _build_transform():
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            # ArcFace standard: normalize về [-1, 1]
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def _face_to_tensor(self, face_bgr: np.ndarray) -> "torch.Tensor":
        """Convert numpy BGR (H,W,3) → normalized tensor (1,3,H,W)."""
        face_rgb = face_bgr[:, :, ::-1].copy()
        pil_img = Image.fromarray(face_rgb.astype(np.uint8))
        return self.transform(pil_img).unsqueeze(0).to(self.device)

    def _face_to_tensor_with_flip(self, face_bgr: np.ndarray) -> "torch.Tensor":
        """
        Trả về batch gồm ảnh gốc và flip ngang → shape (2, 3, H, W).
        Dùng khi đăng ký để tạo anchor embedding mạnh hơn.
        """
        face_rgb = face_bgr[:, :, ::-1].copy()
        pil_img = Image.fromarray(face_rgb.astype(np.uint8))
        t_orig = self.transform(pil_img)

        face_flip = np.fliplr(face_bgr[:, :, ::-1]).copy()
        pil_flip  = Image.fromarray(face_flip.astype(np.uint8))
        t_flip = self.transform(pil_flip)

        return torch.stack([t_orig, t_flip]).to(self.device)

    # ── Public API ────────────────────────────────────────────────────────────

    def get_embedding(self, face_bgr: np.ndarray) -> np.ndarray:
        """
        Trích xuất embedding từ một ảnh khuôn mặt đã align (112×112 BGR).

        Args:
            face_bgr: numpy array HxWx3 BGR — đầu ra của FaceDetector.detect().

        Returns:
            numpy array (512,) — L2-normalized embedding vector.
        """
        tensor = self._face_to_tensor(face_bgr)

        with torch.no_grad():
            embedding = self.model(tensor)
            embedding = F.normalize(embedding, p=2, dim=1)

        return embedding.squeeze().cpu().numpy()

    def get_batch_embeddings(self, faces_bgr: list) -> np.ndarray:
        """
        Trích xuất embedding từ nhiều khuôn mặt cùng lúc (hiệu quả hơn).

        Args:
            faces_bgr: list các numpy array HxWx3 BGR đã align.

        Returns:
            numpy array (N, 512) — mảng các embedding vector đã normalize.
        """
        tensors = []
        for face_bgr in faces_bgr:
            face_rgb = face_bgr[:, :, ::-1].copy()
            pil_img = Image.fromarray(face_rgb.astype(np.uint8))
            tensors.append(self.transform(pil_img))

        batch = torch.stack(tensors).to(self.device)

        with torch.no_grad():
            embeddings = self.model(batch)
            embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings.cpu().numpy()

    def get_mean_embedding(self, faces_bgr: list,
                           use_flip_augmentation: bool = True) -> np.ndarray:
        """
        Tính embedding trung bình từ nhiều ảnh → "Anchor Vector" cho Registration.

        Khi `use_flip_augmentation=True` (mặc định), mỗi khuôn mặt sẽ được
        tính cùng với phiên bản flip ngang. Mean embedding kết quả mạnh hơn
        và ít nhạy cảm với hướng quay đầu nhẹ.

        Args:
            faces_bgr:            list các numpy array HxWx3 BGR đã align.
            use_flip_augmentation: Có dùng flip augmentation không (mặc định True).

        Returns:
            numpy array (512,) — L2-normalized mean embedding.
        """
        if use_flip_augmentation:
            all_tensors = []
            for face_bgr in faces_bgr:
                batch_t = self._face_to_tensor_with_flip(face_bgr)  # (2, 3, H, W)
                all_tensors.append(batch_t)
            batch = torch.cat(all_tensors, dim=0).to(self.device)  # (2N, 3, H, W)

            with torch.no_grad():
                embeddings = self.model(batch)
                embeddings = F.normalize(embeddings, p=2, dim=1)

            emb_np = embeddings.cpu().numpy()
        else:
            emb_np = self.get_batch_embeddings(faces_bgr)

        mean_emb = emb_np.mean(axis=0)
        norm = np.linalg.norm(mean_emb)
        if norm > 0:
            mean_emb = mean_emb / norm
        return mean_emb


# ── Singleton (lazy init) ─────────────────────────────────────────────────────
_extractor_instance = None


def get_extractor() -> EmbeddingExtractor:
    """Trả về singleton EmbeddingExtractor (chỉ load model 1 lần)."""
    global _extractor_instance
    if _extractor_instance is None:
        _extractor_instance = EmbeddingExtractor()
    return _extractor_instance
