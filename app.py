import os
import math
import io
import pandas as pd
import requests
from pathlib import Path
from PIL import Image, ImageDraw
from ultralytics import YOLO
import joblib
import streamlit as st
import urllib.parse
import base64

# إعدادات عامة لواجهة Streamlit
st.set_page_config(
    page_title="نظام اكتشاف حالات الفاقد الكهربائي المحتملة للفئة الزراعية",
    layout="wide",
    page_icon="🌾"
)

API_KEY = "AIzaSyAY7NJrBjS42s6upa9z_qgNLVXESuu366Q"
ZOOM = 16
IMG_SIZE = 640
MAP_TYPE = "satellite"

BASE_DIR = Path(__file__).resolve().parent
IMG_DIR = BASE_DIR / "images"
DETECTED_DIR = BASE_DIR / "DETECTED_FIELDS" / "FIELDS" / "farms"
OUTPUT_FOLDER = BASE_DIR / "output"
MODEL_PATH = str(BASE_DIR / "models" / "best.pt")
ML_MODEL_PATH = BASE_DIR / "models" / "isolation_forest_model.joblib"
SCALER_PATH = BASE_DIR / "models" / "scaler.joblib"

for folder in [IMG_DIR, DETECTED_DIR, OUTPUT_FOLDER]:
    folder.mkdir(parents=True, exist_ok=True)

# تحميل النماذج بشكل صحيح
with st.spinner('تحميل النماذج...'):
    model_yolo = YOLO(MODEL_PATH)
    model_ml = joblib.load(ML_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
st.success("✅ تم تحميل النماذج بنجاح")

# دالة تحميل الصورة من خرائط جوجل
def download_image(lat, lon, meter_id):
    img_path = IMG_DIR / f"{meter_id}.png"
    if img_path.exists():
        return img_path.as_posix()
    params = {
        "center": f"{lat},{lon}",
        "zoom": ZOOM,
        "size": f"{IMG_SIZE}x{IMG_SIZE}",
        "maptype": MAP_TYPE,
        "key": API_KEY
    }
    response = requests.get("https://maps.googleapis.com/maps/api/staticmap", params=params, timeout=20)
    if response.status_code == 200:
        img_path.write_bytes(response.content)
        return img_path.as_posix()
    st.error("خطأ في تحميل الصورة")
    return None

# حساب المساحة من البيكسل إلى متر مربع
def pixel_to_area(lat, box):
    scale = 156543.03392 * abs(math.cos(math.radians(lat))) / (2 ** ZOOM)
    return abs(box[2]-box[0]) * abs(box[3]-box[1]) * scale**2

# كشف الحقول باستخدام نموذج YOLO
def detect_field(img_path, meter_id, info, model):
    img = Image.open(img_path).convert("RGB")
    results = model.predict(source=img_path, conf=0.5)
    if not results or not results[0].boxes:
        return None, None, None
    box = results[0].boxes.xyxy.cpu().numpy()[0]
    area = pixel_to_area(info["y"], box)
    if area < 5000:
        return None, None, None
    ImageDraw.Draw(img).rectangle(box, outline="green", width=3)
    detected_path = DETECTED_DIR / f"{meter_id}.png"
    img.save(detected_path)
    confidence = results[0].boxes.conf.cpu().numpy()[0]
    return round(float(confidence)*100, 2), detected_path.as_posix(), int(area)

# تحديد الأولوية
def determine_priority_custom(meter_id, area, breaker_capacity, consumption):
    if not meter_id.strip(): return "حالة خاصة"
    if area < 10000: return None
    if area > 50000 and (breaker_capacity < 200 or consumption < 20000): return "قصوى"
    if consumption < 10000: return "أولوية عالية"
    return "عادية"

# توليد روابط واتساب وخرائط جوجل
def generate_whatsapp_share_link(meter_id, confidence, area, location_link, quantity, capacity, office_number, priority):
    message = f"""⚡ تقرير حالة عداد زراعي
🔢 {meter_id}
🏢 {office_number}
🚨 {priority}
📊 {confidence}%
🔳 {area:,} م²
💡 {quantity:,} ك.و.س
⚡ {capacity:,} أمبير
📍 {location_link}"""
    return f"https://wa.me/?text={urllib.parse.quote(message)}"

def generate_google_maps_link(lat, lon):
    return f"https://www.google.com/maps?q={lat},{lon}"

# الواجهة الرئيسية
st.markdown("<h2 style='text-align:center;'>🌾 نظام اكتشاف حالات الفاقد الكهربائي الزراعي</h2>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("رفع ملف البيانات (Excel)", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    results = []
    progress_bar = st.progress(0)

    for idx, row in df.iterrows():
        meter_id, lat, lon = str(row["الاشتراك"]), row['y'], row['x']
        office_number = row.get("المكتب", "غير محدد")
        img_path = download_image(lat, lon, meter_id)
        if not img_path: continue
        conf, img_detected, area = detect_field(img_path, meter_id, row, model_yolo)
        if not conf: continue
        priority = determine_priority_custom(meter_id, area, row["Breaker Capacity"], row["الكمية"])
        if not priority: continue
        location_link = generate_google_maps_link(lat, lon)
        whatsapp_link = generate_whatsapp_share_link(meter_id, conf, area, location_link, row['الكمية'], row['Breaker Capacity'], office_number, priority)
        results.append([meter_id, priority, conf, area, whatsapp_link, location_link, img_detected])
        progress_bar.progress((idx+1)/len(df))

    for res in results:
        meter_id, priority, conf, area, whatsapp_link, location_link, img_detected = res
        st.markdown(f"""
        <div>
            <h4>العداد: {meter_id} ({priority})</h4>
            <p>الثقة: {conf}% | المساحة: {area} م²</p>
            <a href="{whatsapp_link}">📲 واتساب</a> | <a href="{location_link}">📍 الموقع</a><br>
            <img src="data:image/png;base64,{base64.b64encode(open(img_detected,"rb").read()).decode()}" width="400">
        </div><hr>
        """, unsafe_allow_html=True)
