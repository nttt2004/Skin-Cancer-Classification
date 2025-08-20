from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# from utils.model import predict_single  # import hàm predict_single
# from utils.model import device


app = Flask(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "utils/skin_resnet50_01.pth"
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ===== Load model =====
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

model = torch.hub.load('pytorch/vision', 'resnet50', pretrained=False)
num_classes = len(checkpoint["class_to_idx"])
model.fc = nn.Linear(model.fc.in_features, num_classes)

model.load_state_dict(checkpoint["model_state_dict"])
model.to(DEVICE)
model.eval()

idx_to_class = {v: k for k, v in checkpoint["class_to_idx"].items()}

# Tiền xử lý ảnh
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

df = pd.read_csv('data\\metadata.csv')

# ====== Từ điển dịch sang tiếng Việt ======
lesion_type_dict = {
    'nv': 'Nốt ruồi',
    'mel': 'U hắc tố',
    'bkl': 'Tổn thương dày sừng lành tính',
    'bcc': 'Ung thư biểu mô tế bào đáy',
    'akiec': 'Dày sừng ánh sáng',
    'vasc': 'Tổn thương mạch máu',
    'df': 'U xơ da'
}

dx_type_dict = {
    'histo': 'Giải phẫu bệnh',
    'follow_up': 'Theo dõi tiến triển',
    'consensus': 'Hội chẩn nhiều chuyên gia',
    'confocal': 'Kính hiển vi confocal'
}

sex_dict = {
    'male': 'Nam',
    'female': 'Nữ'
}

# Tên vị trí thường gặp trong HAM10000 và biến thể
# loc_dict = {
#     'head/neck': 'Đầu / Cổ',
#     'neck': 'Cổ',
#     'torso': 'Thân mình',
#     'trunk': 'Thân mình',
#     'upper extremity': 'Chi trên',
#     'lower extremity': 'Chi dưới',
#     'oral/genital': 'Miệng / Sinh dục',
#     'genital': 'Sinh dục',
#     'palms/soles': 'Lòng bàn tay / Lòng bàn chân',
#     'hand': 'Bàn tay',
#     'foot': 'Bàn chân',
#     'back': 'Lưng',
#     'abdomen': 'Bụng',
#     'chest': 'Ngực',
#     'ear': 'Tai',
#     'face': 'Mặt',
#     'scalp': 'Da đầu',
#     'nose': 'Mũi',
#     'arm': 'Cánh tay',
#     'leg': 'Chân'
# }
loc_dict = {
    'scalp': 'Da đầu',
    'ear': 'Tai',
    'face': 'Mặt',
    'back': 'Lưng',
    'trunk': 'Thân mình',
    'chest': 'Ngực',
    'upper extremity': 'Chi trên',
    'abdomen': 'Bụng',
    'unknown': 'Không xác định',
    'lower extremity': 'Chi dưới',
    'genital': 'Sinh dục',
    'neck': 'Cổ',
    'hand': 'Bàn tay',
    'foot': 'Bàn chân',
    'acral': 'Vùng đầu chi (acral)'
}

# ====== Đọc dữ liệu ======
DF_PATH = 'data/metadata.csv'
if not os.path.exists(DF_PATH):
    raise FileNotFoundError(f"Không tìm thấy file {DF_PATH}")
df_raw = pd.read_csv(DF_PATH)

@app.route('/')
def dashboard():
    # Làm việc trên bản copy để không sửa df_raw gốc
    df = df_raw.copy()

    # Chuẩn hoá cột kiểu chuỗi để tránh lỗi None/NaN
    for col in ['dx', 'dx_type', 'sex', 'localization']:
        if col in df.columns:
            df[col] = df[col].astype(str)

    # Map sang tiếng Việt + điền 'Khác' cho giá trị không khớp/NaN
    if 'dx' in df.columns:
        df['dx'] = df['dx'].map(lesion_type_dict).fillna('Khác')
    else:
        df['dx'] = 'Khác'

    if 'dx_type' in df.columns:
        df['dx_type'] = df['dx_type'].map(dx_type_dict).fillna('Khác')
    else:
        df['dx_type'] = 'Khác'

    if 'sex' in df.columns:
        df['sex'] = df['sex'].map(sex_dict).fillna('Khác')
    else:
        df['sex'] = 'Khác'

    if 'localization' in df.columns:
        df['localization'] = df['localization'].map(loc_dict).fillna('Khác')
    else:
        df['localization'] = 'Khác'

    # --- Biểu đồ Loại bệnh ---
    dx_counts = df['dx'].value_counts(dropna=False)
    dx_labels = dx_counts.index.tolist()
    dx_values = dx_counts.values.tolist()

    # --- Biểu đồ Loại chẩn đoán ---
    dx_type_counts = df['dx_type'].value_counts(dropna=False)
    dx_type_labels = dx_type_counts.index.tolist()
    dx_type_values = dx_type_counts.values.tolist()

    # --- Biểu đồ Giới tính ---
    sex_counts = df['sex'].value_counts(dropna=False)
    sex_labels = sex_counts.index.tolist()
    sex_values = sex_counts.values.tolist()

    # --- Biểu đồ Vị trí ---
    loc_counts = df['localization'].value_counts(dropna=False)
    loc_labels = loc_counts.index.tolist()
    loc_values = loc_counts.values.tolist()

    # --- Biểu đồ Độ tuổi (histogram) ---
    age_series = pd.to_numeric(df.get('age', pd.Series([])), errors='coerce').dropna()
    if len(age_series) > 0:
        bins = list(range(0, 101, 10))  # 0–9, 10–19, ...
        # tạo label [0-9, 10-19, ... , 90-99]
        age_labels = [f"{bins[i]}-{bins[i+1]-1}" for i in range(len(bins)-1)]
        age_hist = pd.cut(age_series, bins=bins, labels=age_labels, right=False).value_counts().sort_index()
        age_values = age_hist.values.tolist()
    else:
        age_labels, age_values = [], []

    # Truyền dữ liệu, luôn đảm bảo là list (không Undefined)
    return render_template(
        'dashboard.html',
        dx_labels=dx_labels or [],
        dx_values=dx_values or [],
        dx_type_labels=dx_type_labels or [],
        dx_type_values=dx_type_values or [],
        sex_labels=sex_labels or [],
        sex_values=sex_values or [],
        loc_labels=loc_labels or [],
        loc_values=loc_values or [],
        age_labels=age_labels or [],
        age_values=age_values or []
    )


@app.route("/gallery")
def gallery():
     # Group images by diagnosis
    gallery_dict = {}
    for dx, group in df.groupby('dx'):
        gallery_dict[dx] = group['image_path'].tolist()[:10]  # show max 6 images per type

    return render_template("gallery.html", gallery_dict=gallery_dict)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "Empty filename"}), 400

        # Lưu file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        uploaded_file_path = os.path.join('uploads', filename).replace("\\", "/")

        # Xử lý ảnh & dự đoán
        image = Image.open(filepath).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            outputs = model(image_tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]

        # Chuẩn bị probabilities
        probabilities = []
        for idx, prob in enumerate(probs):
            class_name = idx_to_class[idx]
            probabilities.append({
                "class": class_name,
                "class_vi": lesion_type_dict.get(class_name, class_name),
                "probability": float(prob)
            })
        probabilities.sort(key=lambda x: -x["probability"])
        prediction = probabilities[0]['class_vi']  # label có xác suất cao nhất

        return render_template('predict.html',
                               uploaded_file=uploaded_file_path,
                               prediction=prediction,
                               predictions=probabilities)

    return render_template('predict.html')

# @app.route('/predict', methods=['GET', 'POST'])
# def predict():
#     if request.method == 'POST':
#         # 1. Kiểm tra file
#         if 'file' not in request.files:
#             return render_template('predict.html', error="Chưa chọn file.")
#         file = request.files['file']
#         if file.filename == '':
#             return render_template('predict.html', error="Chưa chọn file.")

#         # 2. Lưu file
#         filename = secure_filename(file.filename)
#         filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         file.save(filepath)
#         uploaded_file_path = os.path.join('uploads', filename).replace("\\","/")

#         # 3. Lấy metadata từ form
#         age = request.form.get('age')
#         sex = request.form.get('sex', 'unknown')
#         localization = request.form.get('localization', 'unknown')

#         try:
#             age = float(age) if age else None
#         except:
#             age = None

#         # 4. Dự đoán
#         predictions_dict = predict_single(filepath, age=age, sex=sex, localization=localization)

#         # 5. Chuẩn bị mảng probabilities để Chart.js dùng
#         probabilities = []
#         for cls, prob in predictions_dict.items():
#             probabilities.append({
#                 "class": cls,
#                 "class_vi": cls,  # có thể map sang tiếng Việt nếu bạn có dict mapping
#                 "probability": float(prob)
#             })
#         probabilities.sort(key=lambda x: -x["probability"])
#         prediction = probabilities[0]['class_vi']

#         return render_template(
#             'predict.html',
#             uploaded_file=uploaded_file_path,
#             prediction=prediction,
#             predictions=probabilities
#         )

#     return render_template('predict.html')

@app.route("/about")
def about():
    return render_template("about.html", request=request)

if __name__ == "__main__":
    app.run(debug=True)
