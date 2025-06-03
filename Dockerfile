# Sử dụng một base image Python ổn định. Python 3.9-buster thường tốt.
FROM python:3.9-buster

# Đặt biến môi trường để Python không buffer output, giúp xem log dễ hơn
ENV PYTHONUNBUFFERED 1

# Đặt thư mục làm việc bên trong container
WORKDIR /app

# Sao chép file requirements.txt vào thư mục làm việc
COPY requirements.txt .

# Cài đặt các thư viện Python
RUN pip install --no-cache-dir -r requirements.txt

# Cài đặt các thư viện hệ thống mà OpenCV hoặc các thư viện AI khác có thể cần
# Điều này rất quan trọng để tránh lỗi runtime do thiếu shared libraries
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    # Thêm các thư viện khác nếu DeepFace hoặc TensorFlow/Keras yêu cầu
    && rm -rf /var/lib/apt/lists/*

# Sao chép toàn bộ code của ứng dụng (trừ những gì trong .dockerignore)
# vào thư mục làm việc /app trong container
COPY . .

# Expose port mà Flask app của bạn sẽ chạy bên trong container (ví dụ: 5001)
# Port này phải khớp với port bạn cấu hình Flask app chạy trên 0.0.0.0
EXPOSE 5001

# Lệnh để chạy ứng dụng Flask khi container khởi động
# Sử dụng host='0.0.0.0' để Flask app có thể được truy cập từ bên ngoài container qua port đã map
# Thay "app:app" bằng "tên_file_chính:tên_biến_flask_app" nếu khác
# Ví dụ: nếu file chính là main.py và biến app là my_flask_app thì là "main:my_flask_app"
# Đối với file app.py của bạn, CMD sẽ là:
# CMD ["gunicorn", "--bind", "0.0.0.0:5001", "app:app"] # Cho production
CMD ["gunicorn", "--bind", "0.0.0.0:5001", "app:app"]