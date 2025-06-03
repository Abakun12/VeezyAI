# Veezy AI Service (Flask)

Đây là service AI backend được xây dựng bằng Flask cho dự án "Veezy – Website Tổ chức và Bán vé sự kiện trực tuyến tích hợp nhận diện AI"[cite: 1]. Service này cung cấp các API cho việc đăng ký (enrollment) và xác thực (verification) khuôn mặt, sử dụng thư viện DeepFace.

## Tính năng chính

  * **Đăng ký khuôn mặt (`/ai/enroll`):** Nhận một hình ảnh và trả về vector đặc trưng (embedding) của khuôn mặt để lưu trữ.
  * **Xác thực khuôn mặt (`/ai/verify`):** Nhận một hình ảnh trực tiếp và một vector embedding đã lưu trữ, sau đó so sánh để xác thực danh tính.
  * Sử dụng các mô hình AI mạnh mẽ như FaceNet thông qua DeepFace.
  * Có khả năng được đóng gói và triển khai bằng Docker.

## Công nghệ sử dụng

  * **Ngôn ngữ:** Python
  * **Framework:** Flask
  * **Thư viện AI chính:** DeepFace (bao gồm TensorFlow, Keras, OpenCV)
  * **Nhận diện khuôn mặt:** FaceNet (hoặc SFace, tùy cấu hình)
  * **Phát hiện khuôn mặt:** RetinaFace (hoặc các detector khác được DeepFace hỗ trợ)
  * **Deployment (đề xuất):** Docker, Gunicorn

## Cài đặt và Chạy (Sử dụng Docker - Khuyến nghị)

### Yêu cầu

  * Docker đã được cài đặt và đang chạy.

### Các bước

1.  **Clone repository (Nếu dự án của bạn đặt trên Git):**

    ```bash
    git clone <your-repository-url>
    cd Flask_VeezyAI
    ```

2.  **Tạo file `requirements.txt`:**
    Liệt kê các thư viện như `Flask`, `deepface`, `opencv-python`, `numpy`, `gunicorn`, `scikit-learn`.
    Ví dụ:

    ```txt
    Flask
    deepface
    opencv-python
    numpy
    gunicorn
    scikit-learn
    ```

3.  **Tạo file `.dockerignore`:**
    Liệt kê các file/thư mục cần bỏ qua (xem nội dung gợi ý ở các thảo luận trước).

4.  **Tạo file `Dockerfile`:**
    Sử dụng nội dung Dockerfile đã được thảo luận (chọn base image Python, cài đặt dependencies, copy code, expose port, và thiết lập lệnh CMD).

5.  **Build Docker Image:**
    Mở terminal trong thư mục gốc của dự án (`Flask_VeezyAI/`) và chạy:

    ```bash
    docker build -t veezy-ai-service .
    ```

    (Bạn có thể thay `veezy-ai-service` bằng tên image bạn muốn)

6.  **Chạy Docker Container:**

    ```bash
    docker run -d -p 5000:5001 --name veezy-ai-container veezy-ai-service
    ```

      * Lệnh này map port `5000` của máy host tới port `5001` của container (giả sử Gunicorn/Flask trong container chạy ở port `5001`).
      * Thay đổi port mapping nếu cần.

7.  **Kiểm tra:**

      * Service AI sẽ có thể truy cập được tại `http://localhost:5000`.
      * Kiểm tra log của container: `docker logs veezy-ai-container`
      * Kiểm tra các container đang chạy: `docker ps`

## API Endpoints

  * **POST `/ai/enroll`**

      * **Mục đích:** Đăng ký khuôn mặt.
      * **Input:** Form-data với một trường `file` chứa file ảnh.
      * **Output:** JSON chứa `embedding_vector` và các thông tin liên quan.

  * **POST `/ai/verify`**

      * **Mục đích:** Xác thực khuôn mặt.
      * **Input:** Form-data với:
          * Trường `file` chứa file ảnh trực tiếp.
          * Trường `stored_embedding_json` chứa chuỗi JSON của vector embedding đã lưu trữ.
      * **Output:** JSON chứa kết quả `verified` (true/false), `distance`, `threshold`, v.v.

## Hướng Phát Triển Tiếp Theo (Gợi ý)

  * Tích hợp giải pháp Liveness Detection để tăng cường bảo mật.
  * Xây dựng thêm endpoint cho nhận diện 1:N (one-to-many) nếu cần cho các kịch bản check-in không cần định danh trước.
  * Tối ưu hóa hiệu suất và khả năng mở rộng cho môi trường production.
  * Hoàn thiện việc ghi log và giám sát (monitoring).

-----

Đóng góp và phản hồi luôn được chào đón\!
