# Skin Cancer App (Flask + AJAX)

## Cách chạy
```bash
cd skin_cancer_app
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py
```
Mở http://localhost:5000

## Chuẩn bị dataset
Tải dataset từ Kaggle: HAM10000 (kmader/skin-cancer-mnist-ham10000). Đặt file và ảnh như sau:

```
skin_cancer_app/
└── data/
    ├── HAM10000_metadata.csv
    └── images/
        ├── ISIC_0024306.jpg
        ├── ISIC_0024307.jpg
        └── ...
```

> Nếu thiếu dữ liệu, ứng dụng vẫn chạy với dữ liệu giả để minh họa UI, nhưng Gallery có thể trống.

## Ghi chú mô hình
`utils/model.py` hiện dùng mô hình **demo** (ngẫu nhiên) để minh họa tính năng Predict.
Bạn có thể thay bằng model thật (Keras/PyTorch) và cập nhật `load_model()` + `predict_image()` phù hợp.
