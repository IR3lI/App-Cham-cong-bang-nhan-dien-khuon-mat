"""
Vector Database Module — FAISS-backed face identity store.

Lưu trữ:
- face_db.index   : FAISS IndexFlatIP (inner product = cosine sau L2-normalize)
- face_db_meta.pkl: dict {faiss_id → {"employee_id": str, "name": str, "vec": np.ndarray}}

Cả hai file được đặt trong thư mục DB_DIR (gốc dự án).
"""

import os
import sys
import pickle
import numpy as np
from typing import Optional, Tuple, Dict, List

# Buoc stdout/stderr dung UTF-8 de tranh loi UnicodeEncodeError voi tieng Viet tren Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

try:
    import faiss
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False
    print("[VectorDB] CẢNH BÁO: faiss-cpu chưa được cài đặt.")
    print("  Chạy: pip install faiss-cpu")

# ─── Đường dẫn mặc định ──────────────────────────────────────────────────────
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_INDEX_PATH = os.path.join(_BASE_DIR, "face_db.index")
DEFAULT_META_PATH  = os.path.join(_BASE_DIR, "face_db_meta.pkl")

EMBEDDING_DIM  = 512
# Cosine similarity: > 0.35 là khớp (sau L2-normalize, IP = cosine)
COSINE_THRESHOLD = 0.35


class VectorDB:
    """
    FAISS-backed vector database cho nhận diện khuôn mặt.

    - Index: IndexFlatIP (brute-force inner product — chính xác tuyệt đối)
    - Tất cả vector PHẢI được L2-normalize trước khi add/search
    - Cosine similarity = inner product khi vector đã được normalize
    """

    def __init__(
        self,
        index_path: str = DEFAULT_INDEX_PATH,
        meta_path: str = DEFAULT_META_PATH,
        embedding_dim: int = EMBEDDING_DIM,
        threshold: float = COSINE_THRESHOLD,
    ):
        if not _FAISS_AVAILABLE:
            raise ImportError("Cần cài đặt: pip install faiss-cpu")

        self.index_path    = index_path
        self.meta_path     = meta_path
        self.embedding_dim = embedding_dim
        self.threshold     = threshold

        # {faiss_sequential_id: {"employee_id": str, "name": str, "vec": np.ndarray}}
        self._meta: Dict[int, dict] = {}
        self._index: faiss.Index = None

        self._load_or_create()

    # ── Khởi tạo & Persistence ────────────────────────────────────────────────

    def _load_or_create(self):
        """Tải index và metadata từ file, hoặc tạo mới nếu chưa có."""
        index_exists = os.path.exists(self.index_path)
        meta_exists  = os.path.exists(self.meta_path)

        if index_exists and meta_exists:
            self._index = faiss.read_index(self.index_path)
            with open(self.meta_path, "rb") as f:
                raw_meta = pickle.load(f)

            # Validate và chuẩn hoá meta — đảm bảo values là dict Python thuần
            self._meta = {}
            valid = True
            for fid, info in raw_meta.items():
                if not isinstance(info, dict):
                    valid = False
                    break
                if not isinstance(info.get("employee_id"), str):
                    valid = False
                    break
                # Restore vec từ list (nếu đã serialize đúng cách)
                entry = {
                    "employee_id": info["employee_id"],
                    "name": info.get("name", ""),
                }
                if "vec_list" in info:
                    entry["vec"] = np.array(info["vec_list"], dtype=np.float32).reshape(1, -1)
                self._meta[fid] = entry

            if not valid:
                print("[VectorDB] CẢNH BÁO: Meta file bị hỏng — reset database.")
                self._index = faiss.IndexFlatIP(self.embedding_dim)
                self._meta  = {}
            else:
                n = self._index.ntotal
                print(f"[VectorDB] Loaded — {n} faces registered.")
        else:
            self._index = faiss.IndexFlatIP(self.embedding_dim)
            self._meta  = {}
            print("[VectorDB] Created new empty database.")

    def save(self):
        """Ghi index và metadata xuống disk."""
        faiss.write_index(self._index, self.index_path)
        # Serialize meta an toàn: chuyển numpy vec → list Python thuần
        safe_meta = {}
        for fid, info in self._meta.items():
            entry = {
                "employee_id": info["employee_id"],
                "name": info.get("name", ""),
            }
            if "vec" in info and info["vec"] is not None:
                entry["vec_list"] = info["vec"].flatten().tolist()
            safe_meta[fid] = entry
        with open(self.meta_path, "wb") as f:
            pickle.dump(safe_meta, f)
        print(f"[VectorDB] Saved — {self._index.ntotal} faces in database.")

    # ── CRUD ──────────────────────────────────────────────────────────────────

    @property
    def num_faces(self) -> int:
        return self._index.ntotal if self._index else 0

    def list_employees(self) -> List[Dict]:
        """Trả về danh sách nhân viên duy nhất đã đăng ký (không trả về field 'vec')."""
        seen = {}
        for info in self._meta.values():
            eid = info["employee_id"]
            if eid not in seen:
                seen[eid] = {k: v for k, v in info.items() if k != "vec"}
        return list(seen.values())

    def add_face(self, embedding: np.ndarray, employee_id: str, name: str):
        """
        Thêm một embedding vector vào database.

        Args:
            embedding:   numpy (512,) — đã L2-normalize.
            employee_id: mã nhân viên (VD: "NV001").
            name:        tên đầy đủ (VD: "Nguyễn Văn A").
        """
        vec = embedding.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(vec)  # đảm bảo normalize lại

        faiss_id = self._index.ntotal
        self._index.add(vec)
        # Lưu luôn vector vào meta để rebuild index mà không cần faiss internal API
        self._meta[faiss_id] = {
            "employee_id": employee_id,
            "name": name,
            "vec": vec.copy(),  # shape (1, 512)
        }
        print(f"[VectorDB] Added: [{employee_id}] {name} (faiss_id={faiss_id})")

    def remove_employee(self, employee_id: str) -> int:
        """
        Xóa tất cả vector của một nhân viên và rebuild index.
        Trả về số vector đã xóa.
        """
        # Tìm các faiss_id cần xóa
        ids_to_remove = [fid for fid, info in self._meta.items()
                         if info.get("employee_id") == employee_id]
        if not ids_to_remove:
            return 0

        # Rebuild index từ đầu (IndexFlatIP không hỗ trợ xóa trực tiếp)
        # Dùng index.reconstruct(i) — lấy từng vector đơn lẻ, hoạt động trên mọi phiên bản FAISS
        ids_to_remove_set = set(ids_to_remove)
        keep_ids = [i for i in range(self._index.ntotal) if i not in ids_to_remove_set]

        new_index = faiss.IndexFlatIP(self.embedding_dim)
        new_meta  = {}

        for new_id, old_id in enumerate(keep_ids):
            old_info = self._meta[old_id]
            # Ưu tiên dùng vec đã lưu; fallback sang reconstruct(i) cho meta cũ
            if "vec" in old_info:
                vec = old_info["vec"]
            else:
                vec = self._index.reconstruct(old_id).reshape(1, -1)
            new_index.add(vec)
            new_meta[new_id] = {**old_info, "vec": vec}

        self._index = new_index
        self._meta  = new_meta
        print(f"[VectorDB] Removed {len(ids_to_remove)} vectors for [{employee_id}]")
        return len(ids_to_remove)

    # ── Search ────────────────────────────────────────────────────────────────

    def search(
        self, query_embedding: np.ndarray, top_k: int = 1
    ) -> Tuple[Optional[str], Optional[str], float]:
        """
        Tìm nhân viên phù hợp nhất với query embedding.

        Args:
            query_embedding: numpy (512,) — embedding khuôn mặt cần nhận dạng.
            top_k: số kết quả tốt nhất cần lấy.

        Returns:
            employee_id: str hoặc None nếu không khớp.
            name:        str hoặc None.
            similarity:  float (cosine similarity, cao hơn = tốt hơn).
        """
        if self._index.ntotal == 0:
            return None, None, 0.0

        vec = query_embedding.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(vec)

        distances, indices = self._index.search(vec, min(top_k, self._index.ntotal))

        best_dist = float(distances[0][0])
        best_idx  = int(indices[0][0])

        if best_idx < 0 or best_dist < self.threshold:
            # Không đủ tự tin → Unknown
            return None, None, best_dist

        info = self._meta.get(best_idx, {})
        return info.get("employee_id"), info.get("name"), best_dist

    def search_with_scores(
        self, query_embedding: np.ndarray, top_k: int = 3
    ) -> List[Dict]:
        """Trả về top-k kết quả kèm điểm similarity (dùng cho debug)."""
        if self._index.ntotal == 0:
            return []

        vec = query_embedding.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(vec)

        distances, indices = self._index.search(vec, min(top_k, self._index.ntotal))

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue
            info = self._meta.get(int(idx), {})
            results.append({
                "employee_id": info.get("employee_id", "?"),
                "name":        info.get("name", "?"),
                "similarity":  float(dist),
                "matched":     float(dist) >= self.threshold,
            })
        return results


# ── Singleton ─────────────────────────────────────────────────────────────────
_db_instance = None


def get_db() -> VectorDB:
    """Trả về singleton VectorDB."""
    global _db_instance
    if _db_instance is None:
        _db_instance = VectorDB()
    return _db_instance
