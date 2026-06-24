"""
register_faces.py — Script CLI để đăng ký hàng loạt từ thư mục ảnh.
=======================================================================
Dùng khi bạn đã có sẵn ảnh nhân viên trong thư mục employee_db/.

Cấu trúc thư mục:
    employee_db/
    ├── NV001_NguyenVanA/
    │   ├── anh1.jpg
    │   ├── anh2.jpg
    │   └── ...
    └── NV002_TranThiB/
        └── ...

Cách dùng:
    python register_faces.py
    python register_faces.py --db_dir my_employee_db --reset
"""

import os
import sys
import argparse
import glob

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

import cv2
import numpy as np

from core.face_detector       import FaceDetector
from core.embedding_extractor import EmbeddingExtractor
from core.vector_db           import VectorDB

# ─── Cấu hình ────────────────────────────────────────────────────────────────
DEFAULT_DB_DIR    = os.path.join(ROOT_DIR, "employee_db")
SUPPORTED_EXTS    = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
MIN_IMAGES_WARN   = 3   # Cảnh báo nếu ít hơn số này


def parse_args():
    parser = argparse.ArgumentParser(
        description="Đăng ký khuôn mặt nhân viên từ thư mục ảnh vào FAISS database."
    )
    parser.add_argument(
        "--db_dir",
        default=DEFAULT_DB_DIR,
        help=f"Thư mục chứa ảnh nhân viên (mặc định: {DEFAULT_DB_DIR})",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Xóa toàn bộ database cũ trước khi đăng ký mới",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.35,
        help="Ngưỡng cosine similarity (mặc định: 0.35)",
    )
    return parser.parse_args()


def load_images_from_dir(dir_path: str) -> list:
    """Load tất cả ảnh hợp lệ trong một thư mục."""
    images = []
    for ext in SUPPORTED_EXTS:
        images.extend(glob.glob(os.path.join(dir_path, f"*{ext}")))
        images.extend(glob.glob(os.path.join(dir_path, f"*{ext.upper()}")))
    return sorted(set(images))


def process_employee(
    emp_dir: str,
    detector: FaceDetector,
    extractor: EmbeddingExtractor,
) -> tuple:
    """
    Xử lý một thư mục nhân viên:
    1. Đọc ảnh
    2. Detect khuôn mặt
    3. Tính mean embedding
    Trả về (success, employee_id, name, anchor_vector, n_images_used)
    """
    dir_name = os.path.basename(emp_dir)

    # Parse ID và tên từ tên thư mục: "NV001_NguyenVanA"
    parts = dir_name.split("_", 1)
    if len(parts) == 2:
        employee_id, name_raw = parts
        name = name_raw.replace("_", " ")
    else:
        employee_id = dir_name
        name        = dir_name

    image_paths = load_images_from_dir(emp_dir)
    if not image_paths:
        print(f"  ⚠️  Bỏ qua [{dir_name}] — không có ảnh")
        return False, employee_id, name, None, 0

    if len(image_paths) < MIN_IMAGES_WARN:
        print(f"  ⚠️  [{employee_id}] chỉ có {len(image_paths)} ảnh — khuyến nghị ≥{MIN_IMAGES_WARN}")

    valid_faces = []
    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            print(f"    ✗ Không đọc được: {os.path.basename(img_path)}")
            continue

        face_bgr, _, _kps = detector.detect_largest(img)
        if face_bgr is None:
            print(f"    ✗ Không phát hiện mặt: {os.path.basename(img_path)}")
            continue

        valid_faces.append(face_bgr)
        print(f"    ✓ {os.path.basename(img_path)}")

    if not valid_faces:
        print(f"  ❌ [{employee_id}] — Không có ảnh hợp lệ nào!")
        return False, employee_id, name, None, 0

    anchor_vec = extractor.get_mean_embedding(valid_faces)
    return True, employee_id, name, anchor_vec, len(valid_faces)


def main():
    args = parse_args()

    print("=" * 60)
    print("  ĐĂNG KÝ KHUÔN MẶT NHÂN VIÊN (Batch Mode)")
    print("=" * 60)

    # Kiểm tra thư mục DB
    if not os.path.exists(args.db_dir):
        print(f"\n❌ Không tìm thấy thư mục: {args.db_dir}")
        print(f"Hãy tạo thư mục với cấu trúc:")
        print(f"  {args.db_dir}/")
        print(f"  ├── NV001_NguyenVanA/")
        print(f"  │   ├── anh1.jpg")
        print(f"  │   └── anh2.jpg")
        print(f"  └── NV002_TranThiB/")
        sys.exit(1)

    # Lấy danh sách thư mục nhân viên
    emp_dirs = [
        d for d in glob.glob(os.path.join(args.db_dir, "*"))
        if os.path.isdir(d)
    ]

    if not emp_dirs:
        print(f"\n❌ Không tìm thấy thư mục nhân viên nào trong: {args.db_dir}")
        sys.exit(1)

    print(f"\n📁 Tìm thấy {len(emp_dirs)} nhân viên trong: {args.db_dir}")

    # Khởi tạo modules
    print("\n🔄 Đang tải model AI...")
    detector  = FaceDetector()
    extractor = EmbeddingExtractor()
    db        = VectorDB(threshold=args.threshold)

    # Reset nếu cần
    if args.reset:
        import faiss, pickle
        db._index = faiss.IndexFlatIP(512)
        db._meta  = {}
        print("🗑️  Đã xóa database cũ")

    print("\n" + "─" * 60)

    # Xử lý từng nhân viên
    success_count = 0
    fail_count    = 0

    for emp_dir in sorted(emp_dirs):
        dir_name = os.path.basename(emp_dir)
        print(f"\n👤 Đang xử lý: {dir_name}")

        success, emp_id, name, anchor, n_used = process_employee(
            emp_dir, detector, extractor
        )

        if success:
            db.add_face(anchor, emp_id, name)
            print(f"  ✅ [{emp_id}] {name} — đăng ký từ {n_used} ảnh")
            success_count += 1
        else:
            fail_count += 1

    # Lưu database
    print("\n" + "─" * 60)
    print(f"\n💾 Đang lưu database...")
    db.save()

    print("\n" + "=" * 60)
    print(f"  ✅ Thành công: {success_count} nhân viên")
    print(f"  ❌ Thất bại:   {fail_count} nhân viên")
    print(f"  📊 Tổng vector: {db.num_faces}")
    print("=" * 60)
    print("\n✨ Hoàn tất! Chạy ứng dụng: streamlit run app.py")


if __name__ == "__main__":
    main()
