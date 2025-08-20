# Skin-Cancer-Classification  

## Giới thiệu  
Dự án **Skin-Cancer-Classification** nhằm xây dựng hệ thống phân loại bệnh về da bằng cách kết hợp **metadata** và **ảnh da liễu** do người dùng đăng tải.  
Mục tiêu chính:  
- Hỗ trợ nhận diện sớm các bệnh về da.  
- Kết hợp dữ liệu lâm sàng (metadata) và hình ảnh để nâng cao độ chính xác.  
- Ứng dụng các mô hình **Deep Learning** hiện đại.  

## Dataset  
Sử dụng bộ dữ liệu **HAM10000 (Human Against Machine with 10000 training images)**:  
[Kaggle Dataset - Skin Cancer MNIST: HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)  

Bộ dữ liệu bao gồm:  
- **Ảnh da liễu**: hơn 10.000 ảnh về các loại bệnh da.  
- **Metadata**: các thông tin như tuổi (age), giới tính (sex), vị trí tổn thương (localization), nhãn bệnh (dx).  

## Các bước thực hiện  
1. **Xử lý dữ liệu**  
   - Làm sạch metadata (xử lý giá trị thiếu, chuẩn hóa).  
   - Mã hóa dữ liệu phân loại (One-Hot Encoding, Label Encoding).  
   - Tiền xử lý ảnh (resize, normalization, augmentation).  

2. **Xây dựng mô hình**  
   - Mô hình Deep Learning: ResNet18, ResNet50, VGG16, EfficientNet, MobileNet, DenseNet.  
   - Kết hợp metadata + ảnh để huấn luyện mô hình lai.  

3. **Đánh giá mô hình**  
   - Accuracy, Precision, Recall, F1-score.  
   - Confusion Matrix.  

4. **Triển khai**  
   - Cho phép người dùng **upload ảnh** hoặc nhập metadata.  
   - Dự đoán loại bệnh da và hiển thị kết quả.  

## Công nghệ sử dụng  
- **Python, PyTorch, TensorFlow/Keras** cho huấn luyện mô hình.  
- **Pandas, Numpy, Scikit-learn** cho xử lý dữ liệu.  
- **Matplotlib, Seaborn** cho trực quan hóa.  
- **Flask** cho giao diện web/app demo.  

## Hướng phát triển  
- Nâng cao độ chính xác bằng transfer learning.  
- Tích hợp thêm Explainable AI (Grad-CAM) để giải thích kết quả.  
- Xây dựng API phục vụ ứng dụng y tế thực tiễn.  

## Thành viên nhóm  
- Nguyễn Thị Thu Trang  
- Tô Minh Đức
- Nguyễn Minh Đăng
- Nguyễn Trường Giang
- Nguyễn Văn Thi
