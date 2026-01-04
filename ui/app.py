import streamlit as st
import requests
import time

st.set_page_config(layout="wide")
st.title("SecureSurveillance")

VIDEO0 = "http://localhost:8080/video/file0/main"
META0  = "http://localhost:8080/meta/file0/main"

VIDEO1 = "http://localhost:8080/video/file0_1/main"
META1  = "http://localhost:8080/meta/file0_1/main"

# Layout
col_v0, col_v1 = st.columns(2)
col_m0, col_m1 = st.columns(2)

with col_v0:
    st.markdown(f'<img src="{VIDEO0}" style="width:100%"/>', unsafe_allow_html=True)

with col_v1:
    st.markdown(f'<img src="{VIDEO1}" style="width:100%"/>', unsafe_allow_html=True)

meta0_box = col_m0.empty()
meta1_box = col_m1.empty()

# Streamlit-friendly update loop
while st.session_state.get("run", True):
    try:
        meta0 = requests.get(META0, timeout=1).json()
        meta1 = requests.get(META1, timeout=1).json()

        meta0_box.json(meta0)
        meta1_box.json(meta1)

    except Exception as e:
        meta0_box.error(f"Meta0 error: {e}")
        meta1_box.error(f"Meta1 error: {e}")

    time.sleep(0.5)
