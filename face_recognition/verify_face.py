from deepface import DeepFace
import numpy as np
import cv2
import logging
from flask import request, jsonify
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = "Facenet"
DETECTOR_NAME = "retinaface"
DISTANCE_METRIC = "cosine"

def verify_face():
    """
    Endpoint để xác thực khuôn mặt người dùng khi đăng nhập.
    Nhận một file ảnh trực tiếp và một vector embedding đã lưu trữ (dưới dạng chuỗi JSON),
    so sánh chúng bằng FaceNet và trả về kết quả xác thực.
    """
    if 'file' not in request.files:
        logger.warning("Không có file ảnh (live image) nào được cung cấp trong request.")
        return jsonify({"error": "Không có file ảnh trực tiếp nào được cung cấp (No live image file provided)"}), 400

    if 'stored_embedding_json' not in request.form:
        logger.warning("Không có 'stored_embedding_json' nào được cung cấp trong request form.")
        return jsonify({"error": "Không có embedding đã lưu trữ nào được cung cấp (No stored embedding provided)"}), 400

    live_image_file = request.files['file']
    stored_embedding_json_str = request.form['stored_embedding_json']

    if live_image_file.filename == '':
        logger.warning("Tên file ảnh trực tiếp trống.")
        return jsonify({"error": "Chưa chọn file ảnh trực tiếp (No selected live image file)"}), 400

    try:
        # Đọc file ảnh trực tiếp từ request vào bộ nhớ
        live_image_stream = live_image_file.read()
        np_live_image = np.frombuffer(live_image_stream, np.uint8)
        live_img_cv2 = cv2.imdecode(np_live_image, cv2.IMREAD_COLOR)

        if live_img_cv2 is None:
            logger.error("Không thể giải mã hình ảnh trực tiếp.")
            return jsonify({"error": "Không thể giải mã hình ảnh trực tiếp (Could not decode live image)"}), 400

        logger.info(f"Đã nhận và giải mã ảnh trực tiếp: {live_image_file.filename}")

        # Chuyển đổi chuỗi JSON embedding đã lưu trữ thành list Python
        try:
            stored_embedding_vector = json.loads(stored_embedding_json_str)
            if not isinstance(stored_embedding_vector, list):
                raise ValueError("Stored embedding phải là một danh sách (list).")
            # Đảm bảo các phần tử là float (tùy chọn, DeepFace có thể tự xử lý)
            stored_embedding_vector = [float(x) for x in stored_embedding_vector]
            logger.info("Đã chuyển đổi stored_embedding_json thành vector thành công.")
        except json.JSONDecodeError:
            logger.error("Lỗi giải mã JSON cho 'stored_embedding_json'.")
            return jsonify({"error": "Định dạng JSON của embedding đã lưu trữ không hợp lệ (Invalid JSON format for stored embedding)"}), 400
        except ValueError as ve:
            logger.error(f"Lỗi giá trị trong stored_embedding_json: {ve}")
            return jsonify({"error": f"Giá trị không hợp lệ trong embedding đã lưu trữ: {ve}"}), 400


        # --- Bước Xử Lý AI cho Giai đoạn 2: Xác thực ---
        # Sử dụng DeepFace.verify để so sánh ảnh trực tiếp với embedding đã lưu trữ.
        # `DeepFace.verify` sẽ tự động:
        # 1. Phát hiện khuôn mặt trong `live_img_cv2`.
        # 2. Căn chỉnh khuôn mặt (nếu `align=True`).
        # 3. Trích xuất embedding từ `live_img_cv2` bằng `model_name` đã chỉ định.
        # 4. So sánh embedding mới này với `stored_embedding_vector` sử dụng `distance_metric`.

        logger.info(f"Bắt đầu xác thực khuôn mặt bằng mô hình: {MODEL_NAME}, metric: {DISTANCE_METRIC}")

        verification_result = DeepFace.verify(
            img1_path=live_img_cv2, # Ảnh trực tiếp
            img2_path=stored_embedding_vector, # Embedding đã lưu trữ
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_NAME,
            distance_metric=DISTANCE_METRIC,
            align=True,
            enforce_detection=True # Bắt buộc phải tìm thấy khuôn mặt trong ảnh trực tiếp
        )

        # `verification_result` là một dictionary chứa các thông tin như:
        # {
        #  "verified": True/False,
        #  "distance": 0.25, # Khoảng cách giữa hai embeddings
        #  "threshold": 0.40, # Ngưỡng mặc định của model cho metric này
        #  "model": "Facenet",
        #  "detector_backend": "retinaface",
        #  "similarity_metric": "cosine"
        # }
        logger.info(f"Kết quả xác thực: {verification_result}")

        return jsonify({
            "message": "Xác thực hoàn tất (Verification complete)",
            "verified": verification_result.get("verified"),
            "distance": verification_result.get("distance"),
            "threshold": verification_result.get("threshold"),
            "model_details": {
                "model_name": verification_result.get("model"),
                "detector_backend": verification_result.get("detector_backend"),
                "similarity_metric": verification_result.get("similarity_metric")
            }
        }), 200

    except ValueError as ve: # Thường do DeepFace báo lỗi không tìm thấy khuôn mặt trong ảnh trực tiếp
        logger.error(f"Lỗi xử lý khuôn mặt trong quá trình xác thực (ValueError): {str(ve)}")
        if "Face could not be detected" in str(ve) or "multiple faces" in str(ve):
             return jsonify({"error": f"Không thể phát hiện khuôn mặt trong ảnh trực tiếp hoặc phát hiện nhiều hơn một khuôn mặt. (Live image face detection issue: {str(ve)})"}), 400
        return jsonify({"error": f"Lỗi dữ liệu đầu vào khi xác thực: {str(ve)}"}), 400
    except Exception as e:
        logger.exception(f"Lỗi không xác định trong quá trình xác thực: {str(e)}")
        return jsonify({"error": f"Đã xảy ra lỗi nội bộ khi xác thực, vui lòng thử lại sau. (Internal verification error: {str(e)})"}), 500


