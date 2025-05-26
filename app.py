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

# -------------------------
# إعدادات عامة
# -------------------------
st.set_page_config(
    page_title="نظام اكتشاف حالات الفاقد الكهربائي المحتملة للفئة الزراعية",
    layout="wide",
    page_icon="🌾"
)

# إعدادات API للقمر الصناعي
API_KEY = "AIzaSyAY7NJrBjS42s6upa9z_qgNLVXESuu366Q"  # ضع هنا مفتاح Google Maps API الصحيح
ZOOM = 16
IMG_SIZE = 640
MAP_TYPE = "satellite"

# إعدادات المسارات
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

IMG_DIR = BASE_DIR / "images"
DETECTED_DIR = BASE_DIR / "DETECTED_FIELDS" / "FIELDS" / "farms"
MODEL_PATH = str(BASE_DIR / "models" / "best.pt")  # مهم تحويله إلى str
ML_MODEL_PATH = BASE_DIR / "models" / "isolation_forest_model.joblib"
SCALER_PATH = BASE_DIR / "models" / "scaler.joblib"
TEMPLATE_FILE = BASE_DIR / "fram.xlsx"
OUTPUT_FOLDER = BASE_DIR / "output"

IMG_DIR.mkdir(parents=True, exist_ok=True)
DETECTED_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)


# -------------------------
# تحسينات واجهة المستخدم (CSS)
# -------------------------
def setup_ui():
    st.markdown(
        """
        <style>
        :root {
            --high-color: #ff0000;
            --medium-color: #ffa500;
            --low-color: #008000;
        }
        .main {
            direction: rtl;
            text-align: right;
            font-family: 'Arial', sans-serif;
        }
        .header {
            background-color: #2c3e50;
            color: white;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 30px;
            text-align: center;
        }
        .card {
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            padding: 20px;
            margin-bottom: 25px;
            border-left: 5px solid;
            background-color: #f9f9f9;
            page-break-inside: avoid;
        }
        .priority-high {
            border-color: var(--high-color);
            background-color: #ffebee;
        }
        .priority-medium {
            border-color: var(--medium-color);
            background-color: #fff3e0;
        }
        .priority-low {
            border-color: var(--low-color);
            background-color: #e8f5e9;
        }
        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 1px solid #eee;
        }
        .priority-badge {
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 0.9em;
            color: white;
        }
        .high-badge {
            background-color: var(--high-color);
        }
        .medium-badge {
            background-color: var(--medium-color);
        }
        .low-badge {
            background-color: var(--low-color);
        }
        .card-content {
            display: flex;
            gap: 25px;
            align-items: center;
        }
        .card-image-container {
            flex: 1;
            min-width: 300px;
        }
        .card-image {
            width: 100%;
            border-radius: 8px;
            border: 1px solid #ddd;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .card-details {
            flex: 2;
        }
        .detail-row {
            margin-bottom: 10px;
            display: flex;
        }
        .detail-label {
            font-weight: bold;
            min-width: 120px;
            color: #555;
        }
        .detail-value {
            flex: 1;
        }
        .card-actions {
            margin-top: 20px;
            display: flex;
            gap: 15px;
        }
        .action-btn {
            padding: 8px 20px;
            border-radius: 5px;
            text-decoration: none;
            font-weight: bold;
            display: inline-flex;
            align-items: center;
            justify-content: center;
        }
        .whatsapp-btn {
            background: #25D366;
            color: white;
        }
        .map-btn {
            background: #4285F4;
            color: white;
        }
        .progress-container {
            margin: 20px 0;
        }
        .stDataFrame {
            font-size: 0.9em;
        }
        </style>
        """, unsafe_allow_html=True
    )

# -------------------------
# دوال المساعدة
# -------------------------
def download_image(lat, lon, meter_id):
    """
    جلب صورة الأقمار الصناعية من خرائط جوجل وحفظها محليًا.
    """
    img_path = os.path.join(IMG_DIR, f"{meter_id}.png")
    if os.path.exists(img_path):
        return img_path
    base_url = "https://maps.googleapis.com/maps/api/staticmap"
    params = {
        "center": f"{lat},{lon}",
        "zoom": ZOOM,
        "size": f"{IMG_SIZE}x{IMG_SIZE}",
        "maptype": MAP_TYPE,
        "key": API_KEY
    }
    try:
        response = requests.get(base_url, params=params, timeout=20)
        if response.status_code == 200:
            with open(img_path, 'wb') as f:
                f.write(response.content)
            return img_path
    except Exception as e:
        st.error(f"خطأ في تحميل الصورة: {e}")
        return None

def pixel_to_area(lat, box):
    """
    حساب المساحة التقريبية بالاعتماد على إحداثيات البكسل.
    """
    scale = 156543.03392 * abs(math.cos(math.radians(lat))) / (2 ** ZOOM)
    width_m = abs(box[2] - box[0]) * scale
    height_m = abs(box[3] - box[1]) * scale
    return width_m * height_m

def detect_field(img_path, meter_id, info, model):
    """
    اكتشاف الحقول في الصورة:
    - تطبيق نموذج YOLO لتحديد الكائنات من class "field" بثقة >= 50%.
    - اختيار الحقل الأقرب لمركز الصورة (المركز المفترض عند 320,320).
    - التحقق من أن المسافة الفعلية بين مركز الحقل ومركز الصورة <= 300 متر.
    - تجاهل الحالة إذا كانت مساحة الحقل < 10000 م².
    - رسم مربع حول الحقل وإضافة نص معرف.
    """
    try:
        results = model(img_path)
        df_result = results.pandas().xyxy[0]
        # فلترة النتائج: class "field" بثقة >= 50%
        fields = df_result[(df_result["name"] == "field") & (df_result["confidence"] >= 0.50)]
        if fields.empty:
            return None, None, None
        # حساب مركز كل حقل
        fields["center_x"] = (fields["xmin"] + fields["xmax"]) / 2
        fields["center_y"] = (fields["ymin"] + fields["ymax"]) / 2
        fields["dist_center"] = ((fields["center_x"] - 320)**2 + (fields["center_y"] - 320)**2)**0.5
        # اختيار الحقل الأقرب لمركز الصورة
        nearest_field = fields.loc[fields["dist_center"].idxmin()]
        lat_val = info["y"]  # نفترض أن 'y' خط العرض
        scale = 156543.03392 * abs(math.cos(math.radians(lat_val))) / (2 ** ZOOM)
        real_distance_m = nearest_field["dist_center"] * scale
        # شرط المسافة: ≤ 300 متر
        if real_distance_m > 500:
            return None, None, None
        # حساب الثقة والمساحة
        confidence = round(nearest_field["confidence"] * 100, 2)
        image = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        box = [nearest_field["xmin"], nearest_field["ymin"], nearest_field["xmax"], nearest_field["ymax"]]
        draw.rectangle(box, outline="green", width=3)
        area = pixel_to_area(lat_val, box)
        # تجاهل إذا كانت المساحة أقل من 10000 م²
        if area < 5000:
            return None, None, None
        draw.text((10, 10), f"ID: {meter_id}\nArea: {int(area)} m²", fill="yellow")
        center_x, center_y = IMG_SIZE // 2, IMG_SIZE // 2
        radius = 5
        draw.ellipse([(center_x - radius, center_y - radius), (center_x + radius, center_y + radius)],
                     outline="red", width=3)
        os.makedirs(DETECTED_DIR, exist_ok=True)
        out_path = os.path.join(DETECTED_DIR, f"{meter_id}.png")
        image.save(out_path)
        return confidence, out_path, int(area)
    except Exception as e:
        st.error(f"خطأ في معالجة الصورة: {e}")
    return None, None, None

# دالة تنبؤ الشذوذ (Isolation Forest)
def predict_loss(info, model_ml, scaler):
    X = [[info["Breaker Capacity"], info["الكمية"]]]
    X_scaled = scaler.transform(X)
    return model_ml.predict(X_scaled)[0]

# دالة تحديد الأولوية وفق الشروط الجديدة
def determine_priority_custom(meter_id, area, breaker_capacity, consumption):
    """
    - إذا لم يكن هناك رقم عداد (meter_id فارغ): تُصنّف الحالة كـ "حالة خاصة".
    - إذا كانت مساحة الحقل أقل من 10000 م²: لا يتم اعتبارها (تُعاد None).
    - إذا كان الحقل مكتشفًا (توجد قيمة) وكان الاستهلاك أقل من 10000 كيلوواط/ساعة: تُصنّف كـ "أولوية عالية".
    - إذا كان الحقل مكتشفًا وكانت مساحته > 100000 م² وكانت (سعة القاطع < 200 أمبير أو الاستهلاك < 20000 كيلوواط/ساعة): تُصنّف كـ "قصوى".
    - وإلا تُصنّف الحالة كـ "عادية".
    """
    if not meter_id or meter_id.strip() == "":
        return "حالة خاصة"
    # الشرط الأول: مساحة الحقل يجب أن تكون >= 10000 م²، وإلا لا تتم المعالجة.
    if area < 10000:
        return None
    # الشرط القصوى:
    if area > 50000 and (breaker_capacity < 200 or consumption < 20000):
        return "قصوى"
    # الشرط العالي:
    if consumption < 10000:
        return "أولوية عالية"
    # خلاف ذلك:
    return "عادية"

def generate_whatsapp_share_link(meter_id, confidence, area, location_link, quantity, capacity, office_number, priority):
    message = (
        f"⚡ تقرير حالة عداد زراعي\n\n"
        f"🔢 رقم العداد: {meter_id}\n"
        f"🏢 المكتب: {office_number}\n"
        f"🚨 الأولوية: {priority}\n"
        f"📊 ثقة الكشف: {confidence}%\n"
        f"🔳 المساحة: {area:,} م²\n"
        f"💡 الاستهلاك: {quantity:,} ك.و.س\n"
        f"⚡ سعة القاطع: {capacity:,} أمبير\n"
        f"📍 الموقع: {location_link}"
    )
    return f"https://wa.me/?text={urllib.parse.quote(message)}"

def generate_google_maps_link(lat, lon):
    return f"https://www.google.com/maps?q={lat},{lon}"

# -------------------------
# واجهة Streamlit
# -------------------------
setup_ui()

st.markdown("""
<div class="header">
    <h1 style="margin:0;">🌾 نظام اكتشاف حالات الفاقد الكهربائي المحتملة للفئة الزراعية</h1>
</div>
""", unsafe_allow_html=True)

with st.expander("📁 تحميل البيانات", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "📥 تحميل نموذج البيانات",
            open(TEMPLATE_FILE, "rb"),
            file_name="نموذج_البيانات.xlsx",
            help="قم بتحميل نموذج البيانات لملئه بالمعلومات المطلوبة"
        )
    with col2:
        uploaded_file = st.file_uploader(
            "رفع ملف البيانات (Excel)",
            type=["xlsx"],
            help="يرجى رفع ملف Excel يحتوي على بيانات العملاء"
        )

# خيارات الفرز قبل التحليل
sort_columns = ["بدون", "الكمية", "Breaker Capacity", "x", "y"]
st.sidebar.markdown("### خيارات الفرز قبل التحليل")
sort_col = st.sidebar.selectbox("اختر حقل الفرز:", sort_columns)
sort_order = st.sidebar.radio("نوع الفرز:", ["تصاعدي", "تنازلي"], index=0, horizontal=True)

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    df["cont"] = df["الاشتراك"].astype(str).str.strip()
    df["المكتب"] = df["المكتب"].astype(str)
    df["الكمية"] = pd.to_numeric(df["الكمية"], errors="coerce")
    
    if sort_col != "بدون":
        asc_bool = (sort_order == "تصاعدي")
        df = df.sort_values(by=sort_col, ascending=asc_bool)
    
    with st.spinner('جاري تحميل النماذج...'):
        model_yolo = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH, force_reload=True)
        model_ml = joblib.load(ML_MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
    st.success("✅ تم تحميل النماذج بنجاح")
    
    tab1, tab2 = st.tabs(["🎯 النتائج", "📊 البيانات الخام"])
    with tab1:
        st.subheader("النتائج المباشرة")
        results_container = st.container()
        results = []
        gallery = set()
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, row in df.iterrows():
            progress = (idx + 1) / len(df)
            progress_bar.progress(progress)
            status_text.text(f"جاري معالجة السجل {idx + 1} من {len(df)}...")
            
            meter_id = str(row["cont"])
            lat, lon = row['y'], row['x']
            office_number = row["المكتب"]
            
            img_path = download_image(lat, lon, meter_id)
            if not img_path:
                continue
            
            conf, img_detected, area = detect_field(img_path, meter_id, row, model_yolo)
            # إذا لم يتم اكتشاف الحقل أو كانت المساحة غير كافية، نتجاهل السجل
            if conf is None or area is None:
                continue
            
            # تنبؤ نموذج العزل
            anomaly = predict_loss(row, model_ml, scaler)
            # تحديد الأولوية باستخدام الدالة المحدثة: نمرر رقم العداد، والمساحة، وسعة القاطع، والكمية.
            priority = determine_priority_custom(meter_id, area, row["Breaker Capacity"], row["الكمية"])
            # إذا الدالة أعادت None (أي إذا كانت المساحة أقل من 10000)، نتجاهل السجل
            if priority is None:
                continue
            
            result_row = row.copy()
            result_row["نسبة_الثقة"] = conf
            result_row["الأولوية"] = priority
            result_row["المساحة"] = area
            results.append(result_row)
            
            location_link = generate_google_maps_link(lat, lon)
            whatsapp_link = generate_whatsapp_share_link(
                meter_id, conf, area, location_link,
                row['الكمية'], row['Breaker Capacity'], 
                office_number, priority
            )
            
            # تحديد فئة CSS للأولوية
            priority_class = {
                "قصوى": "high",
                "أولوية عالية": "medium",
                "منخفضة": "low",
                "طبيعية": ""
            }.get(priority, "")
            
            with results_container:
                try:
                    with open(img_detected, "rb") as f:
                        img_bytes = f.read()
                    img_base64 = base64.b64encode(img_bytes).decode()
                except Exception as e:
                    st.error(f"خطأ في قراءة الصورة: {e}")
                    img_base64 = ""
                
                st.markdown(f"""
                <div class="card priority-{priority_class}">
                    <div class="card-header">
                        <h3>العداد: {meter_id}</h3>
                        <span class="priority-badge {priority_class}-badge">{priority}</span>
                    </div>
                    <div class="card-content">
                        <div class="card-image-container">
                            <img class="card-image" src="data:image/png;base64,{img_base64}" alt="صورة الحقل">
                        </div>
                        <div class="card-details">
                            <div class="detail-row">
                                <span class="detail-label">المكتب:</span>
                                <span class="detail-value">{office_number}</span>
                            </div>
                            <div class="detail-row">
                                <span class="detail-label">ثقة الكشف:</span>
                                <span class="detail-value">{conf}%</span>
                            </div>
                            <div class="detail-row">
                                <span class="detail-label">المساحة:</span>
                                <span class="detail-value">{area:,} م²</span>
                            </div>
                            <div class="detail-row">
                                <span class="detail-label">الاستهلاك:</span>
                                <span class="detail-value">{row['الكمية']:,} ك.و.س</span>
                            </div>
                            <div class="detail-row">
                                <span class="detail-label">سعة القاطع:</span>
                                <span class="detail-value">{row['Breaker Capacity']:,} أمبير</span>
                            </div>
                            <div class="card-actions">
                                <a href="{whatsapp_link}" class="action-btn whatsapp-btn" target="_blank">مشاركة عبر واتساب</a>
                                <a href="{location_link}" class="action-btn map-btn" target="_blank">عرض على الخريطة</a>
                            </div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        if len(results) == 0:
            st.warning("⚠️ لم يتم العثور على أي نتائج للعرض")
    
    with tab2:
        st.subheader("البيانات الخام")
        st.dataframe(df)
    
    if len(results) > 0:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### تصدير النتائج")
        output_df = pd.DataFrame(results)
        output_excel = io.BytesIO()
        with pd.ExcelWriter(output_excel, engine="xlsxwriter") as writer:
            output_df.to_excel(writer, index=False, sheet_name="النتائج")
        st.sidebar.download_button(
            "📥 تصدير النتائج كملف Excel",
            data=output_excel.getvalue(),
            file_name="نتائج_الفحص.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        st.sidebar.markdown("### 📊 إحصائيات سريعة")
        high_priority = len([r for r in results if r["الأولوية"] == "قصوى"])
        high2 = len([r for r in results if r["الأولوية"] == "أولوية عالية"])
        low_priority = len([r for r in results if r["الأولوية"] == "منخفضة"])
        normal = len([r for r in results if r["الأولوية"] == "طبيعية"])
        st.sidebar.metric("🔴 حالات قصوى", high_priority)
        st.sidebar.metric("🟠 حالات عالية", high2)
        st.sidebar.metric("🟢 حالات منخفضة", low_priority)
        st.sidebar.metric("⚪ حالات طبيعية", normal)
