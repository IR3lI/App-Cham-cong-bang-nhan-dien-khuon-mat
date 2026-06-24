"""
Attendance Logger Module
Ghi và quản lý lịch sử chấm công với:
- File CSV: attendance_log.csv
- Logic Check-In / Check-Out tự động
- Debounce: khóa ID trong N phút sau mỗi lần ghi nhận
"""

import os
import csv
import threading
from datetime import datetime, date, timedelta
from typing import Optional, Dict, List, Tuple

# ─── Cấu hình ────────────────────────────────────────────────────────────────
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_LOG_PATH   = os.path.join(_BASE_DIR, "attendance_log.csv")
DEBOUNCE_MINUTES   = 5       # Khóa N phút sau mỗi lần chấm công

CSV_FIELDNAMES = [
    "Mã Nhân Viên",
    "Tên Nhân Viên",
    "Ngày",
    "Thời gian Check",
    "Trạng thái",
]


class AttendanceLogger:
    """
    Quản lý logic chấm công:
    - Lần đầu trong ngày → "Check-In"
    - Lần tiếp theo → "Check-Out" (cập nhật liên tục cho đến khi rời đi)
    - Debounce: không ghi lại trong N phút sau lần ghi gần nhất

    Thread-safe (dùng Lock).
    """

    def __init__(
        self,
        log_path: str = DEFAULT_LOG_PATH,
        debounce_minutes: int = DEBOUNCE_MINUTES,
    ):
        self.log_path         = log_path
        self.debounce_minutes = debounce_minutes
        self._lock            = threading.Lock()

        # {employee_id: datetime} — thời điểm lần ghi gần nhất
        self._last_seen: Dict[str, datetime] = {}

        # {(employee_id, date): "Check-In" | "Check-Out"} — trạng thái hôm nay
        self._today_status: Dict[Tuple[str, date], str] = {}

        self._ensure_csv_exists()
        print(f"[AttendanceLogger] Log file: {self.log_path}")
        print(f"[AttendanceLogger] Debounce: {self.debounce_minutes} phut")

    # ── Private ───────────────────────────────────────────────────────────────

    def _ensure_csv_exists(self):
        """Tạo file CSV với header nếu chưa tồn tại."""
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
                writer.writeheader()

    def _is_debounced(self, employee_id: str) -> bool:
        """Kiểm tra xem ID có đang bị debounce không."""
        if employee_id not in self._last_seen:
            return False
        elapsed = datetime.now() - self._last_seen[employee_id]
        return elapsed < timedelta(minutes=self.debounce_minutes)

    def _determine_status(self, employee_id: str, today: date) -> str:
        """
        Xác định trạng thái chấm công:
        - Lần đầu hôm nay → "Check-In"
        - Lần sau → "Check-Out"
        """
        key = (employee_id, today)
        current = self._today_status.get(key)

        if current is None:
            return "Check-In"
        elif current == "Check-In":
            return "Check-Out"
        else:
            return "Check-Out"  # Giữ Check-Out cho mọi lần sau

    def _append_to_csv(self, row: dict):
        """Ghi một hàng vào cuối file CSV."""
        with open(self.log_path, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
            writer.writerow(row)

    # ── Public API ────────────────────────────────────────────────────────────

    def record(
        self, employee_id: str, name: str
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Thực hiện chấm công cho một nhân viên.

        Args:
            employee_id: mã nhân viên.
            name:        tên nhân viên.

        Returns:
            (success, status, time_str)
            success:  True nếu ghi thành công, False nếu đang debounce.
            status:   "Check-In" / "Check-Out" / None.
            time_str: chuỗi giờ phút giây / None.
        """
        with self._lock:
            # Kiểm tra debounce
            if self._is_debounced(employee_id):
                return False, None, None

            now   = datetime.now()
            today = now.date()
            status = self._determine_status(employee_id, today)
            time_str = now.strftime("%H:%M:%S")
            date_str = today.strftime("%d/%m/%Y")

            row = {
                "Mã Nhân Viên":   employee_id,
                "Tên Nhân Viên":  name,
                "Ngày":           date_str,
                "Thời gian Check": time_str,
                "Trạng thái":     status,
            }

            self._append_to_csv(row)
            self._last_seen[employee_id]            = now
            self._today_status[(employee_id, today)] = status

            print(f"[Attendance] OK [{employee_id}] {name} -- {status} luc {time_str}")
            return True, status, time_str

    def get_today_records(self) -> List[dict]:
        """
        Đọc tất cả bản ghi chấm công trong ngày hôm nay từ file CSV.

        Returns:
            list các dict với các key theo CSV_FIELDNAMES.
        """
        today_str = date.today().strftime("%d/%m/%Y")
        records   = []

        try:
            with open(self.log_path, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("Ngày") == today_str:
                        records.append(dict(row))
        except FileNotFoundError:
            pass

        return records

    def get_today_summary(self) -> List[dict]:
        """
        Trả về bảng tóm tắt: mỗi nhân viên 1 hàng với Check-In và Check-Out.
        """
        records = self.get_today_records()
        summary: Dict[str, dict] = {}

        for r in records:
            eid = r["Mã Nhân Viên"]
            if eid not in summary:
                summary[eid] = {
                    "Mã NV":    eid,
                    "Tên":      r["Tên Nhân Viên"],
                    "Check-In":  "",
                    "Check-Out": "",
                }
            if r["Trạng thái"] == "Check-In":
                summary[eid]["Check-In"] = r["Thời gian Check"]
            elif r["Trạng thái"] == "Check-Out":
                summary[eid]["Check-Out"] = r["Thời gian Check"]

        return list(summary.values())

    def get_remaining_debounce(self, employee_id: str) -> int:
        """Trả về số giây còn lại của debounce (0 nếu không bị khóa)."""
        if employee_id not in self._last_seen:
            return 0
        elapsed = datetime.now() - self._last_seen[employee_id]
        remaining = timedelta(minutes=self.debounce_minutes) - elapsed
        return max(0, int(remaining.total_seconds()))


# ── Singleton ─────────────────────────────────────────────────────────────────
_logger_instance = None


def get_logger() -> AttendanceLogger:
    """Trả về singleton AttendanceLogger."""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = AttendanceLogger()
    return _logger_instance
