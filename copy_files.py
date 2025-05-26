import shutil
from pathlib import Path

# مسار المشروع القديم (عدله إذا اختلف)
OLD_PROJECT_DIR = Path(r"C:\Users\Sec\Documents\DEEP")

# مسار المشروع الجديد (نفس مسار هذا السكريبت الحالي)
NEW_PROJECT_DIR = Path(__file__).resolve().parent

# ملفات المصدر والمجلدات الهدف
files_to_copy = {
    OLD_PROJECT_DIR / "app.py": NEW_PROJECT_DIR / "app.py",
    OLD_PROJECT_DIR / "requirements.txt": NEW_PROJECT_DIR / "requirements.txt",
    OLD_PROJECT_DIR / "fram.xlsx": NEW_PROJECT_DIR / "fram.xlsx",
    OLD_PROJECT_DIR / "isolation_forest_model.joblib": NEW_PROJECT_DIR / "models/isolation_forest_model.joblib",
    OLD_PROJECT_DIR / "scaler.joblib": NEW_PROJECT_DIR / "models/scaler.joblib",
    OLD_PROJECT_DIR / "yolov5/farms_project/field_detector/weights/best.pt": NEW_PROJECT_DIR / "models/best.pt"
}

# تأكد من وجود المجلدات الهدف
for dest_path in files_to_copy.values():
    dest_path.parent.mkdir(parents=True, exist_ok=True)

# نسخ الملفات
for src_path, dest_path in files_to_copy.items():
    if src_path.exists():
        shutil.copy(src_path, dest_path)
        print(f"✅ تم نسخ الملف بنجاح: {src_path.name}")
    else:
        print(f"❌ الملف غير موجود ولم يتم نسخه: {src_path}")

print("\n🎉 تم الانتهاء من النسخ بنجاح!")
