import streamlit as st
import requests
import math
import time

st.set_page_config(layout="wide")
st.title("SecureSurveillance")

# ----------------------------
# Helpers
# ----------------------------
def get_streams(base_url: str, timeout=1.0):
    r = requests.get(f"{base_url}/streams", timeout=timeout)
    r.raise_for_status()
    return r.json()  # ["file0_0/main", "file0_1/main"]

def split_stream_id(s: str):
    # "file0_0/main" -> ("file0_0", "main")
    if "/" not in s:
        return s, "main"
    src, channel = s.split("/", 1)
    return src, channel

def compose_urls(base: str, stream_id: str):
    src, channel = split_stream_id(stream_id)
    base = base.rstrip("/")
    video = f"{base}/video/{src}/{channel}"
    meta  = f"{base}/meta/{src}/{channel}"
    return video, meta

def safe_get_json(url: str, timeout: float):
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.header("Backend")

base_url = st.sidebar.text_input(
    "Base URL",
    value="http://localhost:8080"
)

refresh_ms = st.sidebar.slider(
    "Refresh interval (ms)", 200, 2000, 500, step=100
)

timeout_s = st.sidebar.slider(
    "HTTP timeout (s)", 0.2, 3.0, 0.7, step=0.1
)

cols = st.sidebar.slider(
    "Grid columns", 1, 4, 2
)

show_meta = st.sidebar.checkbox("Show metadata", True)
show_links = st.sidebar.checkbox("Show stream links", False)

run = st.sidebar.toggle("Run", value=True)

# ----------------------------
# Load streams from backend
# ----------------------------
try:
    streams = get_streams(base_url, timeout=timeout_s)
except Exception as e:
    st.error(f"Failed to fetch /streams: {e}")
    st.stop()

if not streams:
    st.warning("No streams available.")
    st.stop()

# ----------------------------
# Stream selector
# ----------------------------
selected = st.multiselect(
    "Select streams",
    options=streams,
    default=streams[: min(2, len(streams))],
)

if not selected:
    st.info("Select at least one stream.")
    st.stop()

# ----------------------------
# Auto refresh (Streamlit-safe)
# ----------------------------
if run:
    time.sleep(refresh_ms / 1000.0)
    st.rerun()

# ----------------------------
# Render grid
# ----------------------------
n = len(selected)
rows = math.ceil(n / cols)

for r in range(rows):
    row_streams = selected[r * cols : (r + 1) * cols]
    grid = st.columns(cols)

    for c, stream_id in enumerate(row_streams):
        with grid[c]:
            video_url, meta_url = compose_urls(base_url, stream_id)

            st.subheader(stream_id)

            if show_links:
                st.code(video_url)
                st.code(meta_url)

            # MJPEG stream
            st.markdown(
                f'<img src="{video_url}" style="width:100%; height:auto; border-radius:10px;" />',
                unsafe_allow_html=True,
            )

            if show_meta:
                try:
                    meta = safe_get_json(meta_url, timeout_s)
                    st.json(meta)
                except Exception as e:
                    st.warning(f"Meta unavailable: {e}")
