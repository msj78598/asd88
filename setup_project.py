from pathlib import Path

# حدد مسار المشروع الحالي تلقائيًا
BASE_DIR = Path(__file__).resolve().parent

# هيكلة المجلدات المطلوبة
directories = [
    BASE_DIR / "models",
    BASE_DIR / "output",
    BASE_DIR / "images",
    BASE_DIR / "DETECTED_FIELDS",
]

# إنشاء المجلدات
for directory in directories:
    directory.mkdir(parents=True, exist_ok=True)

print("✅ تم إنشاء جميع المجلدات بنجاح")
