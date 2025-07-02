# Giai đoạn 1: Build - Cài đặt các thư viện
# Sử dụng một image Python 3.10 đầy đủ để có các công cụ build cần thiết
FROM python:3.10-slim as builder

# Thiết lập thư mục làm việc
WORKDIR /app

# Cài đặt các thư viện hệ thống cần thiết cho OpenCV, etc.
# Chạy bước này trước để tận dụng cache nếu không thay đổi
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Tăng tốc độ cài đặt pip
RUN pip install --upgrade pip

# Sao chép file requirements.txt trước
COPY requirements.txt .

# Cài đặt các thư viện Python vào một thư mục riêng
RUN pip wheel --no-cache-dir --wheel-dir /app/wheels -r requirements.txt

# Giai đoạn 2: Final - Tạo image cuối cùng để chạy
# Sử dụng cùng một base image gọn nhẹ để đảm bảo tương thích
FROM python:3.10-slim

# Tạo một user không phải root để tăng cường bảo mật
RUN useradd --create-home appuser
WORKDIR /home/appuser

# Cài đặt các thư viện hệ thống cần thiết cho runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Sao chép các thư viện đã được cài đặt từ giai đoạn build
COPY --from=builder /app/wheels /wheels
COPY --from=builder /app/requirements.txt .
RUN pip install --no-cache /wheels/*

# Sao chép toàn bộ code của ứng dụng
COPY . .

# Chuyển quyền sở hữu cho user mới
RUN chown -R appuser:appuser /home/appuser

# Chuyển sang user mới
USER appuser

# Mở port mà ứng dụng sẽ chạy
EXPOSE 8080

# Biến môi trường để Gunicorn biết chạy trên port nào
ENV PORT 8080

# Lệnh để khởi chạy ứng dụng bằng Gunicorn
# --bind: Chạy trên tất cả các địa chỉ IP trên port được cung cấp
# --workers: Số lượng process xử lý (điều chỉnh tùy theo CPU)
# --threads: Số lượng thread trên mỗi worker
# --timeout: Thời gian tối đa cho một request (đặt 0 để không giới hạn, hữu ích cho các tác vụ AI dài)
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--threads", "8", "--timeout", "0", "wsgi:app"]
