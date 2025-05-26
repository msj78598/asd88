import os
import math
import io
import pandas as pd
import requests
from pathlib import Path
from PIL import Image, ImageDraw
import torch
import joblib
from sklearn.preprocessing import StandardScaler
import streamlit as st
import urllib.parse
import base64
import numpy as np
import sys

# ------------------------- إعدادات عامة -------------------------
st.set_page_config(
    page_title="نظام اكتشاف حالات الفاقد الكهربائي المحتملة للفئة الزراعية",
    layout="wide",
    page_icon="🌾"
)

API_KEY = "AIzaSyAY7NJrBjS42s6upa9z_qgNLVXESuu366Q"
ZOOM = 16
IMG_SIZE = 640
MAP_TYPE = "satellite"

# ------------------------- إعدادات المسارات -------------------------
BASE_DIR = Path(__file__).resolve().parent

IMG_DIR = BASE_DIR / "images"
DETECTED_DIR = BASE_DIR / "DETECTED_FIELDS" / "FIELDS" / "farms"
MODEL_PATH = (BASE_DIR / "models" / "best.pt").as_posix()
ML_MODEL_PATH = (BASE_DIR / "models" / "isolation_forest_model.joblib").as_posix()
SCALER_PATH = (BASE_DIR / "models" / "scaler.joblib").as_posix()
TEMPLATE_FILE = (BASE_DIR / "fram.xlsx").as_posix()
OUTPUT_FOLDER = BASE_DIR / "output"

for folder in [IMG_DIR, DETECTED_DIR, OUTPUT_FOLDER]:
    folder.mkdir(parents=True, exist_ok=True)

# ------------------------- CSS -------------------------
def setup_ui():
    st.markdown("""<style>
    :root {--high-color: #ff0000;--medium-color: #ffa500;--low-color: #008000;}
    .main {direction: rtl;text-align: right;font-family: 'Arial';}
    .header {background-color: #2c3e50;color: white;padding: 15px;border-radius: 10px;margin-bottom: 30px;text-align: center;}
    .card {border-radius: 10px;box-shadow: 0 4px 8px rgba(0,0,0,0.1);padding: 20px;margin-bottom: 25px;border-left: 5px solid;background-color: #f9f9f9;}
    .priority-high {border-color: var(--high-color);background-color: #ffebee;}
    .priority-medium {border-color: var(--medium-color);background-color: #fff3e0;}
    .priority-low {border-color: var(--low-color);background-color: #e8f5e9;}
    </style>""", unsafe_allow_html=True)

# ------------------------- دوال مساعدة -------------------------
def download_image(lat, lon, meter_id):
    img_path = IMG_DIR / f"{meter_id}.png"
    if img_path.exists():
        return img_path.as_posix()
    params = {"center": f"{lat},{lon}", "zoom": ZOOM, "size": f"{IMG_SIZE}x{IMG_SIZE}", "maptype": MAP_TYPE, "key": API_KEY}
    response = requests.get("https://maps.googleapis.com/maps/api/staticmap", params=params, timeout=20)
    if response.status_code == 200:
        img_path.write_bytes(response.content)
        return img_path.as_posix()
    st.error(f"خطأ في تحميل الصورة: {response.status_code}")
    return None

def pixel_to_area(lat, box):
    scale = 156543.03392 * abs(math.cos(math.radians(lat))) / (2 ** ZOOM)
    return abs(box[2]-box[0]) * abs(box[3]-box[1]) * scale**2

def detect_field(img_path, meter_id, info, model):
    img = Image.open(img_path).convert("RGB")
    results = model(img_path)
    df_result = results.pandas().xyxy[0]
    fields = df_result[(df_result["name"]=="field")&(df_result["confidence"]>=0.50)]
    if fields.empty: return None,None,None
    nearest_field = fields.iloc[0]
    confidence = round(nearest_field["confidence"]*100,2)
    box = [nearest_field["xmin"], nearest_field["ymin"], nearest_field["xmax"], nearest_field["ymax"]]
    area = pixel_to_area(info["y"], box)
    if area < 5000: return None,None,None
    ImageDraw.Draw(img).rectangle(box, outline="green", width=3)
    detected_path = DETECTED_DIR / f"{meter_id}.png"
    img.save(detected_path)
    return confidence, detected_path.as_posix(), int(area)

def predict_loss(info, model_ml, scaler):
    X = [[info["Breaker Capacity"], info["الكمية"]]]
    return model_ml.predict(scaler.transform(X))[0]

def determine_priority_custom(meter_id, area, breaker_capacity, consumption):
    if not meter_id.strip(): return "حالة خاصة"
    if area<10000: return None
    if area>50000 and (breaker_capacity<200 or consumption<20000): return "قصوى"
    if consumption<10000: return "أولوية عالية"
    return "عادية"

def generate_whatsapp_share_link(meter_id, confidence, area, location_link, quantity, capacity, office_number, priority):
    message = f"⚡ تقرير حالة عداد زراعي\n🔢 {meter_id}\n🏢 {office_number}\n🚨 {priority}\n📊 {confidence}%\n🔳 {area:,} م²\n💡 {quantity:,} ك.و.س\n⚡ {capacity:,} أمبير\n📍 {location_link}"
    return f"https://wa.me/?text={urllib.parse.quote(message)}"

def generate_google_maps_link(lat, lon):
    return f"https://www.google.com/maps?q={lat},{lon}"

# ------------------------- الواجهة -------------------------
setup_ui()
st.markdown("""<div class="header"><h1>🌾 نظام اكتشاف حالات الفاقد الكهربائي المحتملة للفئة الزراعية</h1></div>""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("رفع ملف البيانات (Excel)", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    with st.spinner('تحميل النماذج...'):
        sys.path.insert(0, (BASE_DIR/"yolov5").as_posix())
        from models.common import DetectMultiBackend
        model_yolo = DetectMultiBackend(MODEL_PATH, device='cpu')
        model_ml, scaler = joblib.load(ML_MODEL_PATH), joblib.load(SCALER_PATH)
    st.success("✅ تم تحميل النماذج بنجاح")
    
    for idx, row in df.iterrows():
        meter_id, lat, lon = str(row["الاشتراك"]), row['y'], row['x']
        img_path = download_image(lat, lon, meter_id)
        if not img_path: continue
        conf, img_detected, area = detect_field(img_path, meter_id, row, model_yolo)
        if not conf: continue
        priority = determine_priority_custom(meter_id, area, row["Breaker Capacity"], row["الكمية"])
        if not priority: continue
        location_link = generate_google_maps_link(lat, lon)
        whatsapp_link = generate_whatsapp_share_link(meter_id, conf, area, location_link, row['الكمية'], row['Breaker Capacity'], row["المكتب"], priority)
        st.markdown(f"""
        <div class="card priority-{priority}">
        <h3>العداد: {meter_id}</h3> ثقة: {conf}%، المساحة: {area} م²<br>
        <a href="{whatsapp_link}">واتساب</a> | <a href="{location_link}">خريطة</a>
        <img src="data:image/png;base64,{base64.b64encode(open(img_detected,"rb").read()).decode()}" width="100%">
        </div>""",unsafe_allow_html=True)
