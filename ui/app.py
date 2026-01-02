import streamlit as st
import requests
import time

st.set_page_config(layout="wide")
st.title("SecureSurveillance")

VIDEO = "http://localhost:8080/video"
META = "http://localhost:8080/meta"

st.markdown(f'<img src = "{VIDEO}" style = "width:100%; height:auto" />', unsafe_allow_html=True)

box = st.empty()
while True:
    try:
        box.json(requests.get(META, timeout=  1).json())
    except Exception as e:
        box.write(f'meta error: {e}')
    time.sleep(0.5)