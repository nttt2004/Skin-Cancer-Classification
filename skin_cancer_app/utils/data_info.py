import os
import pandas as pd

# Expected dataset layout (user should place files accordingly):
# data/
#   HAM10000_metadata.csv
#   images/
#       <image_id>.jpg

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
META_CSV = os.path.join(DATA_DIR, "HAM10000_metadata.csv")
IMG_DIR  = os.path.join(DATA_DIR, "images")

CLASS_NAME_MAP = {
    "akiec": "Actinic keratoses / intraepithelial carcinoma",
    "bcc":   "Basal cell carcinoma",
    "bkl":   "Benign keratosis-like lesions",
    "df":    "Dermatofibroma",
    "mel":   "Melanoma",
    "nv":    "Melanocytic nevi",
    "vasc":  "Vascular lesions",
}

def _read_metadata():
    if os.path.exists(META_CSV):
        df = pd.read_csv(META_CSV)
        return df
    # Fallback placeholder dataframe
    columns = ["image_id", "dx"]
    sample = pd.DataFrame([
        {"image_id": "placeholder_1", "dx": "nv"},
        {"image_id": "placeholder_2", "dx": "mel"},
        {"image_id": "placeholder_3", "dx": "bkl"},
    ], columns=columns)
    return sample

def dataset_ready_message(info):
    if info.get("meta_found") and info.get("images_found"):
        return "Dataset đã sẵn sàng ✅"
    if not info.get("meta_found") and not info.get("images_found"):
        return "Chưa tìm thấy metadata và thư mục ảnh. Hãy tải dataset từ Kaggle và đặt vào thư mục data/."
    if not info.get("meta_found"):
        return "Thiếu file HAM10000_metadata.csv trong thư mục data/."
    if not info.get("images_found"):
        return "Thiếu thư mục data/images chứa ảnh JPG của dataset."
    return ""

def get_dataset_info():
    df = _read_metadata()
    meta_found = os.path.exists(META_CSV)
    images_found = os.path.exists(IMG_DIR) and any(name.lower().endswith(".jpg") for name in os.listdir(IMG_DIR))    
    total_images = len(df)
    class_counts = df["dx"].value_counts().to_dict() if "dx" in df.columns else {}
    return {
        "total_images": int(total_images),
        "classes": class_counts,
        "class_names": CLASS_NAME_MAP,
        "meta_found": meta_found,
        "images_found": images_found,
        "meta_path": META_CSV,
        "img_dir": IMG_DIR,
    }

def get_gallery_images(limit=8):
    df = _read_metadata()
    by_class = {}
    # Collect up to 'limit' images per class
    for dx in sorted(df["dx"].unique()):
        subset = df[df["dx"] == dx].head(limit)
        paths = []
        for img_id in subset["image_id"].tolist():
            jpg_path = os.path.join(IMG_DIR, f"{img_id}.jpg")
            if os.path.exists(jpg_path):
                paths.append(jpg_path.replace("\\", "/"))
        by_class[dx] = paths
    return by_class
