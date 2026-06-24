# Hướng Dẫn Cài Đặt & Sử Dụng — Hệ Thống Chấm Công Khuôn Mặt

## Cấu trúc dự án

```
du_an_cham_cong/
├── models/
│   ├── __init__.py
│   └── iresnet.py              ← Kiến trúc IResNet-50 (ArcFace compatible)
├── core/
│   ├── __init__.py
│   ├── face_detector.py        ← MTCNN face detection
│   ├── embedding_extractor.py  ← Load .pth, extract 512-d embedding
│   └── vector_db.py            ← FAISS IndexFlatIP (cosine similarity)
├── attendance/
│   ├── __init__.py
│   └── logger.py               ← CSV logging + debounce + Check-In/Out logic
├── ui/
│   ├── __init__.py
│   └── streamlit_app.py        ← Giao diện chính (Dual Mode)
├── app.py                      ← Entry point: streamlit run app.py
├── register_faces.py           ← CLI batch registration từ thư mục ảnh
├── requirements.txt
├── ir50_scface_best_arcface_unfreze_all.pth  ← File trọng số của bạn
│
│   (Auto-generated sau khi chạy)
├── face_db.index               ← FAISS index (binary)
├── face_db_meta.pkl            ← Metadata nhân viên (pickle)
└── attendance_log.csv          ← Lịch sử chấm công
```

---

## Bước 1: Cài đặt môi trường

### Tạo môi trường ảo (khuyến nghị)
```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

### Cài đặt thư viện
```powershell
pip install -r requirements.txt
```

> **Lưu ý GPU**: Nếu có NVIDIA GPU, thay `faiss-cpu` bằng `faiss-gpu` trong requirements.txt
> và cài PyTorch với CUDA: https://pytorch.org/get-started/locally/

---

## Bước 2: Khởi động ứng dụng

```powershell
streamlit run app.py
```

Trình duyệt sẽ tự động mở tại `http://localhost:8501`

---

## Bước 3: Đăng ký nhân viên mới (Chế độ Đăng ký)

### Cách A: Qua Camera trực tiếp (Live Enrollment) — KHUYẾN NGHỊ

1. Trên sidebar trái → nhấn nút **"➕ Đăng ký"**
2. Điền **Mã Nhân Viên** (VD: `NV001`) và **Tên Nhân Viên** (VD: `Nguyễn Văn A`)
3. Chọn số frame thu thập (mặc định 20)
4. Nhấn **"🎥 Bắt đầu thu thập"**
5. Nhân viên đứng trước camera, nhìn thẳng — chờ thanh tiến trình đầy
6. Hệ thống tự tính Mean Embedding và lưu vào database

### Cách B: Từ thư mục ảnh có sẵn (Batch)

1. Tạo cấu trúc thư mục:
```
employee_db/
├── NV001_NguyenVanA/
│   ├── anh1.jpg
│   └── anh2.jpg
└── NV002_TranThiB/
    └── anh1.jpg
```

2. Chạy script đăng ký hàng loạt:
```powershell
python register_faces.py
```

---

## Bước 4: Sử dụng Chế độ Chấm công

1. Trên sidebar → nhấn **"📋 Chấm công"** (mặc định khi mở app)
2. Tick checkbox **"▶ Bật camera"**
3. Nhân viên đứng trước camera:
   - **Khung xanh lá** = nhận ra, đang ghi chấm công
   - **Khung đỏ** = Unknown
4. Thông báo nổi xuất hiện khi chấm công thành công
5. Bảng chấm công bên phải tự cập nhật

---

## Cài đặt có thể điều chỉnh (trên Sidebar)

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| Ngưỡng Cosine | 0.35 | Cao hơn = chặt hơn (0.3–0.5 cho ArcFace) |
| Debounce | 5 phút | Thời gian khóa giữa 2 lần chấm |
| Camera ID | 0 | 0 = webcam mặc định, 1+ = camera ngoài |

---

## Logic chấm công

```
Lần 1 trong ngày → Check-In
Lần 2+ trong ngày → Check-Out
Trong vòng 5 phút sau lần cuối → BỎ QUA (debounce)
```

**File log** `attendance_log.csv`:
```
Mã Nhân Viên,Tên Nhân Viên,Ngày,Thời gian Check,Trạng thái
NV001,Nguyễn Văn A,13/06/2026,08:15:32,Check-In
NV001,Nguyễn Văn A,13/06/2026,17:45:10,Check-Out
```

---

## Xử lý lỗi thường gặp

### `FileNotFoundError: ir50_scface_best_arcface_unfreze_all.pth`
→ Đảm bảo file `.pth` nằm trong thư mục `du_an_cham_cong/`

### `RuntimeError: Error(s) in loading state_dict`
→ Kiến trúc mô hình không khớp. Kiểm tra lại xem model có phải IResNet-50 không.
→ Có thể thử `iresnet100()` trong `core/embedding_extractor.py`

### Camera không mở được
→ Thử thay đổi Camera ID từ 0 → 1 → 2 trên sidebar
→ Kiểm tra quyền truy cập camera của trình duyệt/Python

### Nhận diện kém / Nhiều Unknown
→ Giảm ngưỡng Cosine (0.25–0.30)
→ Đăng ký lại với nhiều frame hơn (25–30)
→ Đảm bảo ánh sáng tốt khi đăng ký

---

## Xuất dữ liệu

- Nhấn **"⬇ Tải attendance_log.csv"** trong sidebar để xuất báo cáo
- File CSV có thể mở trực tiếp bằng Excel
