"""
app.py — Entry point: streamlit run app.py

Chay: streamlit run app.py
"""
# File nay la entry point chinh cua Streamlit.
# Streamlit se chay truc tiep noi dung cua file nay.
# Toan bo UI duoc dinh nghia trong ui/streamlit_app.py
# va duoc include o day bang exec de Streamlit nhan biet dung cach.

import os
import sys

# Buoc stdout/stderr dung UTF-8 de tranh UnicodeEncodeError voi tieng Viet tren Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Dam bao thu muc goc du an nam trong sys.path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Chay noi dung cua streamlit_app.py trong cung namespace nay
# (Streamlit can biet entry point file thuc su)
_app_path = os.path.join(ROOT_DIR, "ui", "streamlit_app.py")
with open(_app_path, "r", encoding="utf-8") as _f:
    exec(compile(_f.read(), _app_path, "exec"), {"__file__": _app_path, "__name__": "__main__"})
