"""
Hệ thống Chấm công bằng Khuôn mặt — Giao diện Streamlit
=========================================================
Chế độ 1: Attendance Mode  — nhận diện & chấm công thời gian thực
Chế độ 2: Registration Mode — đăng ký khuôn mặt mới qua camera

Chạy: streamlit run app.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time
import threading
import queue
from datetime import datetime

import cv2
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu

from core.face_detector       import get_detector
from core.embedding_extractor import get_extractor
from core.vector_db           import get_db
from attendance.logger        import get_logger

# ─── Cấu hình trang ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hệ thống Chấm công Khuôn mặt",
    page_icon="face_id",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS tùy chỉnh ───────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: #e2e8f0;
}

/* Minimal dark background */
.stApp {
    background-color: #0f1115;
    min-height: 100vh;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #1a1c23;
    border-right: 1px solid rgba(255,255,255,0.05);
}

section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #f1f5f9 !important;
}

/* Force all sidebar text to be bright */
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div {
    color: #cbd5e1 !important;
}

/* Main content text */
.stMarkdown p, .stMarkdown li, .stMarkdown span {
    color: #e2e8f0;
}

/* Checkbox, radio label */
[data-testid="stCheckbox"] label p,
[data-testid="stRadio"] label p {
    color: #e2e8f0 !important;
}

/* Warning / info box text */
[data-testid="stAlert"] p {
    color: #1e293b !important;
}

/* Header card */
.header-card {
    background-color: #1a1c23;
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 8px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.5rem;
}

.header-card h1 {
    font-size: 1.6rem;
    font-weight: 600;
    color: #f8fafc;
    margin: 0;
}

.header-card p {
    color: #94a3b8;
    margin: 0.3rem 0 0;
    font-size: 0.9rem;
}

/* Mode badge */
.mode-badge-attendance {
    display: inline-block;
    background-color: rgba(16, 185, 129, 0.1);
    color: #34d399;
    padding: 0.2rem 0.8rem;
    border-radius: 4px;
    font-size: 0.8rem;
    font-weight: 500;
    border: 1px solid rgba(16, 185, 129, 0.2);
}

.mode-badge-registration {
    display: inline-block;
    background-color: rgba(245, 158, 11, 0.1);
    color: #fbbf24;
    padding: 0.2rem 0.8rem;
    border-radius: 4px;
    font-size: 0.8rem;
    font-weight: 500;
    border: 1px solid rgba(245, 158, 11, 0.2);
}

/* Notification card */
.notif-card {
    background-color: #1a1c23;
    border-left: 4px solid #10b981;
    border-radius: 4px;
    padding: 1rem 1.5rem;
    margin: 0.75rem 0;
    animation: fadeSlideIn 0.3s ease-out;
}

.notif-card .notif-name {
    font-size: 1.1rem;
    font-weight: 600;
    color: #f8fafc;
}

.notif-card .notif-detail {
    color: #94a3b8;
    font-size: 0.85rem;
    margin-top: 0.2rem;
}

/* Unknown notification */
.notif-unknown {
    background-color: #1a1c23;
    border-left: 4px solid #ef4444;
    border-radius: 4px;
    padding: 0.75rem 1.5rem;
    margin: 0.5rem 0;
    color: #f8fafc;
    font-size: 0.9rem;
}

/* Stats card */
.stat-card {
    background-color: #1a1c23;
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 8px;
    padding: 1rem;
    text-align: center;
}

.stat-card .stat-num {
    font-size: 1.8rem;
    font-weight: 600;
    color: #f8fafc;
}

.stat-card .stat-label {
    font-size: 0.75rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* Progress bar container */
.progress-container {
    background-color: #1a1c23;
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 8px;
    padding: 1.2rem;
    margin: 0.75rem 0;
}

.progress-label {
    color: #e2e8f0;
    font-size: 0.9rem;
    font-weight: 500;
    margin-bottom: 0.5rem;
}

/* Attendance table */
.attendance-table {
    background-color: #1a1c23;
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 8px;
    overflow: hidden;
}

/* Animation */
@keyframes fadeSlideIn {
    from { opacity: 0; transform: translateY(-5px); }
    to   { opacity: 1; transform: translateY(0); }
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}

.recording-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    background-color: #ef4444;
    border-radius: 50%;
    animation: pulse 1.2s ease-in-out infinite;
    margin-right: 8px;
}

/* Streamlit element overrides */
div[data-testid="stDataFrame"] {
    background: transparent;
}

.stButton > button {
    border-radius: 6px;
    font-weight: 500;
    transition: all 0.2s;
    background-color: #272a35;
    border: 1px solid rgba(255,255,255,0.1);
}

.stButton > button:hover {
    background-color: #3f4354;
    border: 1px solid rgba(255,255,255,0.2);
    color: white;
}
</style>
""", unsafe_allow_html=True)


# ─── Session State khởi tạo ───────────────────────────────────────────────────
def init_session():
    defaults = {
        "mode":               "attendance",
        "notification":       None,
        "reg_step":           "idle",
        "reg_faces":          [],
        "reg_emp_id":         "",
        "reg_emp_name":       "",
        "reg_num_frames":     20,
        "camera_running":     False,
        "last_refresh":       time.time(),
        "unknown_detected":   False,
        "reg_target":         20,
        "reg_submitted":      False,
        "reg_countdown_start": None,   # thời điểm bắt đầu đếm ngược
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()


# ─── Load resources (cached) ─────────────────────────────────────────────────
@st.cache_resource(show_spinner="Đang tải model AI...")
def load_resources():
    detector  = get_detector()
    extractor = get_extractor()
    db        = get_db()
    logger    = get_logger()
    return detector, extractor, db, logger

detector, extractor, db, logger = load_resources()


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Hệ thống Chấm công")
    st.markdown("---")

    # Chế độ hoạt động
    selected_mode = option_menu(
        menu_title="Chế độ",
        options=["Chấm công", "Đăng ký"],
        icons=["camera-video", "person-plus"],
        menu_icon="menu-app",
        default_index=0 if st.session_state.mode == "attendance" else 1,
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"color": "#94a3b8", "font-size": "16px"}, 
            "nav-link": {"font-size": "14px", "text-align": "left", "margin":"0px", "--hover-color": "#272a35", "color": "#cbd5e1"},
            "nav-link-selected": {"background-color": "#3b82f6", "color": "white", "font-weight": "500"},
        }
    )

    if selected_mode == "Chấm công" and st.session_state.mode != "attendance":
        st.session_state.mode = "attendance"
        st.session_state.reg_step = "idle"
        st.session_state.reg_faces = []
        st.session_state.notification = None
        st.rerun()
    elif selected_mode == "Đăng ký" and st.session_state.mode != "registration":
        st.session_state.mode = "registration"
        st.session_state.reg_step = "idle"
        st.session_state.reg_faces = []
        st.rerun()

    st.markdown("---")

    # Thống kê database
    st.markdown("### Thống kê DB")
    employees = db.list_employees()
    today_records = logger.get_today_summary()

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-num">{len(employees)}</div>
            <div class="stat-label">Nhân Viên</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-num">{len(today_records)}</div>
            <div class="stat-label">Đã Chấm</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Cài đặt
    st.markdown("### Cài đặt")

    # Threshold điều chỉnh được
    similarity_threshold = st.slider(
        "Ngưỡng nhận diện (Cosine)",
        min_value=0.20, max_value=0.80, value=0.40, step=0.05,
        help="Thấp hơn = dễ nhận ra hơn (có thể nhầm). Cao hơn = chặt hơn (có thể bỏ sót).\n"
             "Khuyến nghị: 0.35–0.50 với webcam thông thường."
    )
    db.threshold = similarity_threshold

    debounce = st.slider(
        "Debounce (phút)",
        min_value=1, max_value=30, value=5,
        help="Thời gian khóa sau mỗi lần chấm công thành công"
    )
    logger.debounce_minutes = debounce

    show_debug = st.checkbox(
        "Hiện debug score",
        value=True,
        help="Hiển thị similarity score trên frame camera — giúp căn chỉnh ngưỡng"
    )
    if "show_debug" not in st.session_state or show_debug != st.session_state.get("show_debug_prev"):
        st.session_state["show_debug"] = show_debug
        st.session_state["show_debug_prev"] = show_debug

    # Cố định camera laptop (ID = 0)
    camera_id = 0

    st.markdown("---")

    # Quản lý nhân viên
    if employees:
        st.markdown("### Nhân viên đã đăng ký")
        for emp in employees:
            col_e1, col_e2 = st.columns([4, 1])
            with col_e1:
                st.markdown(f"<div style='padding-top: 5px; color:#cbd5e1'>"
                            f"<strong style='color:#f8fafc'>{emp['employee_id']}</strong> — {emp['name']}</div>",
                            unsafe_allow_html=True)
            with col_e2:
                if st.button("Xóa", key=f"del_{emp['employee_id']}",
                             help=f"Xóa {emp['name']}"):
                    removed = db.remove_employee(emp["employee_id"])
                    db.save()
                    st.success(f"Đã xóa {removed} vector của {emp['name']}")
                    st.rerun()


# ─── Header ───────────────────────────────────────────────────────────────────
mode_label = "Chế độ Chấm công" if st.session_state.mode == "attendance" else "Chế độ Đăng ký"
mode_class = "mode-badge-attendance" if st.session_state.mode == "attendance" else "mode-badge-registration"

st.markdown(f"""
<div class="header-card">
    <h1>Hệ thống Chấm công Khuôn mặt</h1>
    <p>IResNet-50 · ArcFace · FAISS Vector Database &nbsp;
        <span class="{mode_class}">{mode_label}</span>
        &nbsp;&middot;&nbsp;
        <span style="color:#64748b;font-size:0.75rem;">RetinaFace · Face Alignment · CLAHE</span>
    </p>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  CHẾ ĐỘ ĐĂNG KÝ (REGISTRATION MODE)
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state.mode == "registration":

    col_form, col_cam = st.columns([1, 2])

    with col_form:
        st.markdown("### Thông tin nhân viên")

        # — Ô nhập liệu (không dùng st.form để hiện validation inline) —
        emp_id = st.text_input(
            "Mã Nhân Viên *",
            placeholder="VD: NV001",
            value=st.session_state.reg_emp_id,
            key="input_emp_id",
            disabled=(st.session_state.reg_step == "collecting"),
        )
        # Lỗi mã nhân viên
        if st.session_state.reg_submitted and not emp_id.strip():
            st.markdown(
                "<p style='color:#ef4444;font-size:0.82rem;margin-top:-12px;'>&#9888; Vui lòng nhập mã nhân viên!</p>",
                unsafe_allow_html=True,
            )

        emp_name = st.text_input(
            "Tên Nhân Viên *",
            placeholder="VD: Nguyễn Văn A",
            value=st.session_state.reg_emp_name,
            key="input_emp_name",
            disabled=(st.session_state.reg_step == "collecting"),
        )
        # Lỗi tên nhân viên
        if st.session_state.reg_submitted and not emp_name.strip():
            st.markdown(
                "<p style='color:#ef4444;font-size:0.82rem;margin-top:-12px;'>&#9888; Vui lòng nhập tên nhân viên!</p>",
                unsafe_allow_html=True,
            )

        num_frames = st.slider(
            "Số frame thu thập", 10, 30,
            value=st.session_state.reg_num_frames,
            disabled=(st.session_state.reg_step == "collecting"),
        )

        start_btn = st.button(
            "▶️ Bắt đầu thu thập",
            use_container_width=True,
            disabled=(st.session_state.reg_step == "collecting"),
        )

        if start_btn:
            st.session_state.reg_submitted = True
            st.session_state.reg_num_frames = num_frames
            if emp_id.strip() and emp_name.strip():
                st.session_state.reg_emp_id         = emp_id.strip()
                st.session_state.reg_emp_name        = emp_name.strip()
                st.session_state.reg_target          = num_frames
                st.session_state.reg_step            = "countdown"
                st.session_state.reg_faces           = []
                st.session_state.reg_countdown_start = time.time()

        # Trạng thái hiện tại
        step = st.session_state.reg_step
        if step == "idle":
            st.markdown("""
            <div style="background: linear-gradient(135deg, rgba(59,130,246,0.15), rgba(16,185,129,0.1));
                        border: 1px solid rgba(59,130,246,0.4);
                        border-left: 4px solid #3b82f6;
                        border-radius: 8px; padding: 1rem 1.2rem; margin-top: 0.5rem;">
                <div style="color:#93c5fd; font-weight:600; font-size:0.9rem; margin-bottom:0.3rem;">
                    📸 Hướng dẫn đăng ký
                </div>
                <div style="color:#cbd5e1; font-size:0.85rem; line-height:1.6;">
                    Điền đầy đủ Mã và Tên nhân viên, sau đó nhấn
                    <strong style="color:#fbbf24">▶️ Bắt đầu thu thập</strong>
                    để mở camera — hệ thống sẽ đếm ngược 5 giây rồi tự động ghi lại khuôn mặt.
                </div>
            </div>
            """, unsafe_allow_html=True)
        elif step == "collecting":
            collected = len(st.session_state.reg_faces)
            target    = st.session_state.reg_target
            pct       = min(collected / target, 1.0)
            st.markdown(f"""
            <div class="progress-container">
                <div class="progress-label">
                    <span class="recording-dot"></span>
                    Đang thu thập: {collected}/{target} frame
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.progress(pct)
        elif step == "processing":
            st.info("Đang tính toán embedding vector...")
        elif step == "done":
            st.success(f"Đã đăng ký thành công {st.session_state.reg_emp_name}!")
            st.markdown(f"""
            <div class="notif-card">
                <div class="notif-name">{st.session_state.reg_emp_name}</div>
                <div class="notif-detail">Mã NV: {st.session_state.reg_emp_id} · Đã lưu vào database</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Đăng ký người tiếp theo"):
                st.session_state.reg_step      = "idle"
                st.session_state.reg_faces     = []
                st.session_state.reg_emp_id    = ""
                st.session_state.reg_emp_name  = ""
                st.session_state.reg_submitted = False
                st.rerun()
        elif step == "error":
            st.error("Không phát hiện khuôn mặt! Vui lòng thử lại.")
            if st.button("Thử lại"):
                st.session_state.reg_step      = "idle"
                st.session_state.reg_faces     = []
                st.session_state.reg_submitted = False

    with col_cam:
        st.markdown("### Camera")
        frame_placeholder = st.empty()
        guide_placeholder = st.empty()

        if st.session_state.reg_step in ("countdown", "collecting", "processing"):
            guide_placeholder.markdown("""
            <div style="background-color:#1a1c23;border:1px solid rgba(255,255,255,0.05);
                        border-radius:8px;padding:0.75rem 1rem;margin-bottom:0.5rem;color:#94a3b8;">
                Nhìn thẳng vào camera · Ánh sáng đầy đủ · Không đeo kính tối
            </div>
            """, unsafe_allow_html=True)

            if st.session_state.reg_step in ("countdown", "collecting"):
                cap = cv2.VideoCapture(int(camera_id))
                if not cap.isOpened():
                    st.error(f"❌ Không thể mở camera ID={camera_id}. Hãy kiểm tra:"
                             f"\n- Camera có đang bị ứng dụng khác chiếm dụng không?"
                             f"\n- Cắm lại hoặc kiểm tra driver camera.")
                    st.session_state.reg_step = "idle"
                    st.session_state.reg_faces = []
                else:
                    try:
                        COUNTDOWN_SECS = 5
                        while st.session_state.reg_step in ("countdown", "collecting"):
                            ret, frame = cap.read()
                            if not ret:
                                st.error("Mất kết nối camera trong quá trình thu thập!")
                                st.session_state.reg_step = "idle"
                                break

                            display = frame.copy()

                            # ── COUNTDOWN (hiện số đếm ngược, không thu thập) ──
                            if st.session_state.reg_step == "countdown":
                                elapsed = time.time() - (st.session_state.reg_countdown_start or time.time())
                                remaining = int(COUNTDOWN_SECS - elapsed) + 1
                                remaining = max(1, min(remaining, COUNTDOWN_SECS))

                                # Lớp phủ tối nhẹ
                                overlay = display.copy()
                                cv2.rectangle(overlay, (0, 0), (display.shape[1], display.shape[0]), (0, 0, 0), -1)
                                cv2.addWeighted(overlay, 0.35, display, 0.65, 0, display)

                                # Số đếm ngược to giánh giữ́a
                                h, w = display.shape[:2]
                                txt = str(remaining)
                                font = cv2.FONT_HERSHEY_DUPLEX
                                scale = 5.0
                                thickness = 12
                                (tw, th), _ = cv2.getTextSize(txt, font, scale, thickness)
                                cx = (w - tw) // 2
                                cy = (h + th) // 2
                                # Bóng
                                cv2.putText(display, txt, (cx + 4, cy + 4), font, scale, (0, 0, 0), thickness + 4)
                                # Chữ chính màu vàng
                                cv2.putText(display, txt, (cx, cy), font, scale, (0, 200, 255), thickness)

                                # Dòng nhắc
                                msg = "Chuan bi..."
                                cv2.putText(display, msg, (w // 2 - 120, cy + 70),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)

                                frame_rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
                                frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

                                if elapsed >= COUNTDOWN_SECS:
                                    st.session_state.reg_step = "collecting"
                                continue   # không xử lý landmark / thu thập

                            # ── COLLECTING ──
                            face_bgr, box, kps = detector.detect_largest(frame)

                            if face_bgr is not None and box is not None:
                                x1, y1, x2, y2 = box
                                collected = len(st.session_state.reg_faces)
                                target    = st.session_state.reg_target
                                progress  = collected / target

                                # Màu thanh tiến trình: xanh lam nhạt (blue)
                                color = (250, 150, 50)  # BGR -> (50, 150, 250) in RGB

                                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)

                                # Thanh tiến trình trên frame
                                bar_x1, bar_y1 = x1, y2 + 5
                                bar_x2         = x1 + int((x2 - x1) * progress)
                                bar_y2         = y2 + 15
                                cv2.rectangle(display, (x1, y2+5), (x2, y2+15), (50,50,50), -1)
                                if bar_x2 > bar_x1:
                                    cv2.rectangle(display, (bar_x1, bar_y1), (bar_x2, bar_y2), color, -1)

                                # Text đếm frame
                                cv2.putText(
                                    display,
                                    f"{collected}/{target}",
                                    (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2,
                                )

                                # Thu thập frame
                                if collected < target:
                                    st.session_state.reg_faces.append(face_bgr.copy())

                                # Vẽ landmarks (nếu có)
                                if kps is not None:
                                    for px, py in kps.astype(int):
                                        cv2.circle(display, (px, py), 3, (0, 255, 255), -1)
                            else:
                                cv2.putText(
                                    display,
                                    "Khong phat hien khuon mat",
                                    (30, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2,
                                )

                            frame_rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
                            frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

                            if len(st.session_state.reg_faces) >= st.session_state.reg_target:
                                st.session_state.reg_step = "processing"
                                break

                    finally:
                        cap.release()

            # Xử lý embedding sau khi thu thập đủ
            if st.session_state.reg_step == "processing":
                faces = st.session_state.reg_faces
                if len(faces) == 0:
                    st.session_state.reg_step = "error"
                else:
                    with st.spinner("Đang tính Mean Embedding..."):
                        anchor_vec = extractor.get_mean_embedding(faces)
                    db.add_face(
                        anchor_vec,
                        st.session_state.reg_emp_id,
                        st.session_state.reg_emp_name,
                    )
                    db.save()
                    st.session_state.reg_step  = "done"
                    st.session_state.reg_faces = []
                st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
#  CHẾ ĐỘ CHẤM CÔNG (ATTENDANCE MODE)
# ═══════════════════════════════════════════════════════════════════════════════
else:
    col_video, col_info = st.columns([3, 2])

    with col_video:
        st.markdown("### Camera thời gian thực")

        if db.num_faces == 0:
            st.warning("""
            **Database trống!** Chưa có nhân viên nào được đăng ký.
            Chuyển sang chế độ Đăng ký (trên sidebar) để thêm nhân viên.
            """)

        run_camera = st.checkbox("Bật camera", value=False, key="cam_toggle")
        frame_placeholder = st.empty()
        notif_placeholder = st.empty()

        if run_camera:
            cap = cv2.VideoCapture(int(camera_id))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

            if not cap.isOpened():
                st.error(f"Không thể mở camera ID={camera_id}. Kiểm tra kết nối!")
            else:
                frame_count = 0
                PROCESS_EVERY = 3  # Xử lý AI mỗi N frame (giảm tải CPU)

                while run_camera:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Mất kết nối camera!")
                        break

                    frame_count += 1
                    display = frame.copy()

                    # Chỉ chạy AI mỗi PROCESS_EVERY frame
                    if frame_count % PROCESS_EVERY == 0:
                        faces_bgr, boxes, lm_list = detector.detect(frame)

                        for i, (face_bgr, box) in enumerate(zip(faces_bgr, boxes)):
                            x1, y1, x2, y2 = box
                            embedding = extractor.get_embedding(face_bgr)

                            # Lấy top-3 để debug
                            top_results = db.search_with_scores(embedding, top_k=3)
                            emp_id, name, similarity = db.search(embedding)

                            if emp_id and name:
                                # Nhân viên nhận ra
                                color = (100, 200, 50)   # xanh lá BGR
                                label = f"{name} ({similarity:.2f})"
                                cv2.rectangle(display, (x1, y1), (x2, y2), color, 3)
                                cv2.rectangle(display, (x1, y1-35), (x2, y1), color, -1)
                                cv2.putText(display, label, (x1+5, y1-10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

                                # Ghi chấm công
                                success, status, time_str = logger.record(emp_id, name)
                                if success:
                                    st.session_state.notification = {
                                        "name":      name,
                                        "emp_id":    emp_id,
                                        "status":    status,
                                        "time":      time_str,
                                        "timestamp": time.time(),
                                    }
                            else:
                                # Unknown — hiển thị score tốt nhất để debug
                                best_score = top_results[0]["similarity"] if top_results else 0.0
                                best_name  = top_results[0]["name"]       if top_results else "?"
                                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 0, 220), 3)
                                cv2.rectangle(display, (x1, y1-35), (x2, y1), (0, 0, 220), -1)
                                if st.session_state.get("show_debug", True) and top_results:
                                    unknown_label = f"? {best_name[:8]} {best_score:.2f}"
                                else:
                                    unknown_label = "Unknown"
                                cv2.putText(display, unknown_label, (x1+5, y1-10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)

                            # Vẽ landmarks (5 points)
                            if lm_list and i < len(lm_list) and lm_list[i] is not None:
                                for px, py in lm_list[i].astype(int):
                                    cv2.circle(display, (px, py), 3, (0, 215, 255), -1)

                    # Timestamp trên frame
                    ts = datetime.now().strftime("%H:%M:%S  %d/%m/%Y")
                    cv2.putText(display, ts, (10, display.shape[0]-15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

                    frame_rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

                    # Hiện notification (tự ẩn sau 4 giây)
                    notif = st.session_state.get("notification")
                    if notif and (time.time() - notif["timestamp"]) < 4:
                        notif_placeholder.markdown(f"""
                        <div class="notif-card">
                            <div class="notif-name">Xin chào, {notif['name']}!</div>
                            <div class="notif-detail">
                                {notif['status']} thành công lúc <strong>{notif['time']}</strong>
                                &nbsp;·&nbsp; Mã NV: {notif['emp_id']}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        notif_placeholder.empty()

                    time.sleep(0.03)  # ~30 FPS tối đa

                cap.release()
        else:
            # Camera tắt — hiện placeholder
            frame_placeholder.markdown("""
            <div style="height:400px; background-color:#1a1c23;
                        border:1px solid rgba(255,255,255,0.05);
                        border-radius:8px; display:flex;
                        align-items:center; justify-content:center;
                        color:#64748b; font-size:1rem;">
                Bật camera để bắt đầu chấm công
            </div>
            """, unsafe_allow_html=True)

    with col_info:
        st.markdown("### Chấm công hôm nay")

        # Nút refresh
        if st.button("Cập nhật bảng"):
            st.rerun()

        today_summary = logger.get_today_summary()

        if not today_summary:
            st.markdown("""
            <div style="background-color:#1a1c23;border:1px dashed rgba(255,255,255,0.1);
                        border-radius:8px;padding:2rem;text-align:center;
                        color:#64748b;font-size:0.95rem;">
                Chưa có ai chấm công hôm nay
            </div>
            """, unsafe_allow_html=True)
        else:
            import pandas as pd
            df = pd.DataFrame(today_summary)
            st.dataframe(
                df,
                width="stretch",
                hide_index=True,
                column_config={
                    "Mã NV":    st.column_config.TextColumn("Mã NV", width="small"),
                    "Tên":      st.column_config.TextColumn("Họ & Tên"),
                    "Check-In": st.column_config.TextColumn("Check-In", width="small"),
                    "Check-Out":st.column_config.TextColumn("Check-Out", width="small"),
                },
            )

            # Thống kê nhanh
            total = len(today_summary)
            checked_out = sum(1 for r in today_summary if r.get("Check-Out"))
            st.markdown(f"""
            <div style="display:flex;gap:0.75rem;margin-top:0.75rem;">
                <div class="stat-card" style="flex:1">
                    <div class="stat-num" style="color:#3b82f6">{total}</div>
                    <div class="stat-label">Tổng có mặt</div>
                </div>
                <div class="stat-card" style="flex:1">
                    <div class="stat-num" style="color:#10b981">{total - checked_out}</div>
                    <div class="stat-label">Đang làm việc</div>
                </div>
                <div class="stat-card" style="flex:1">
                    <div class="stat-num" style="color:#f59e0b">{checked_out}</div>
                    <div class="stat-label">Đã về</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # Export CSV
        st.markdown("### Xuất dữ liệu")
        log_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "attendance_log.csv"
        )
        if os.path.exists(log_path):
            with open(log_path, "rb") as f:
                st.download_button(
                    label="Tải attendance_log.csv",
                    data=f,
                    file_name=f"attendance_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    width="stretch",
                )
