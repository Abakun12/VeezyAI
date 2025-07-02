from deepface import DeepFace
import numpy as np
import cv2
import logging
from flask import request, jsonify


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = "Facenet"
DETECTOR_NAME = "retinaface"

def enroll_face():
    """
    Enroll face image to database.
    """
    if request.method == 'POST':
        if 'file' not in request.files:
            logger.warning("Không có trường 'file' trong request.files cho POST request.")
            return jsonify({"error": "Trường 'file' không có trong request"}), 400

        file = request.files['file']
        if file.filename == '':
            logger.warning("Không có file nào được chọn để upload.")
            return jsonify({"error": "Không có file nào được chọn"}), 400

        try:
            images_stream= file.read()
            np_image = np.frombuffer(images_stream, np.uint8)
            img_cv2 = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
            # images_stream = "img/avarta.png"
            # img_cv2 = cv2.imread(images_stream)

            if img_cv2 is None:
                logger.warning("Error reading image")
                return jsonify({"error": "Error reading image"}), 400

            logger.info(f"Image read successfully: {file.filename}")

            logger.info(f"Start extraction with specific model: {MODEL_NAME}, detector: {DETECTOR_NAME}")
            embedding_objects = DeepFace.represent(
                img_path=img_cv2,
                model_name=MODEL_NAME,
                detector_backend=DETECTOR_NAME,
                enforce_detection=True,
                anti_spoofing=True,
                align=True
            )

            if not embedding_objects or not isinstance(embedding_objects, list) or not embedding_objects[0]:
                logger.warning("Error extracting embedding")
                return jsonify({"error": "Error extracting embedding"}), 400

            first_face_obj = embedding_objects[0]
            embedding_vector = first_face_obj['embedding']
            facial_area = first_face_obj['facial_area']
            face_confidence = first_face_obj.get('face_confidence', None)

            logger.info(f"Trích xuất đặc trưng thành công. Khu vực khuôn mặt: {facial_area}, Độ tin cậy: {face_confidence}")

            response_data = {
                "message": "Đăng ký khuôn mặt thành công (Enrollment successful)",
                "embedding": embedding_vector,
                "facial_area_detected": facial_area,
                "model_details": {
                    "model_name": MODEL_NAME,
                    "detector_backend": DETECTOR_NAME
                }
            }
            if face_confidence is not None:
                response_data["face_confidence"] = face_confidence

            return jsonify(response_data), 200

        except ValueError as ve:
            error_message = str(ve).lower()  # Chuyển thông báo lỗi về chữ thường để dễ kiểm tra
            logger.error(f"Lỗi xử lý khuôn mặt (ValueError): {error_message}")

            # Kiểm tra xem thông báo lỗi có chứa manh mối về giả mạo không
            if 'spoof' in error_message or 'real-time face liveness' in error_message:
                logger.warning("Cảnh báo giả mạo: Khuôn mặt có thể không phải người thật.")
                return jsonify({"error": "Phát hiện hành vi đáng ngờ. Vui lòng sử dụng ảnh chụp trực tiếp."}), 403
            else:
                # Nếu là lỗi ValueError khác (ví dụ: không tìm thấy khuôn mặt)
                return jsonify({"error": f"Lỗi phát hiện khuôn mặt: {str(ve)}"}), 400
        except Exception as e:
            logger.exception(f"Lỗi không xác định: {str(e)}")
            return jsonify({"error": f"Đã xảy ra lỗi nội bộ: {str(e)}"}), 500
    else:
        # Xử lý cho các method khác nếu bạn cho phép (ví dụ: trả về lỗi 405)
        return jsonify({"error": "Phương thức không được phép cho endpoint này. Vui lòng sử dụng POST."}), 405

def preload_models():
    """Hàm để tải trước các mô hình AI."""
    try:
        logger.info(f"Đang tải mô hình AI ({MODEL_NAME}) và detector ({DETECTOR_NAME})...")
        DeepFace.build_model(MODEL_NAME)
        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
        # Sử dụng extract_faces thay vì detectFace (đã deprecated)
        # và đặt enforce_detection=False khi chỉ dùng để tải trước model với ảnh giả.
        DeepFace.extract_faces(
            img_path=dummy_img,
            detector_backend=DETECTOR_NAME,
            enforce_detection=False,
            anti_spoofing=True
        )
        logger.info(f"Các mô hình AI ({MODEL_NAME}, {DETECTOR_NAME}) đã sẵn sàng.")
    except Exception as e:
        logger.exception(f"Lỗi khi tải mô hình: {e}")

def extract_embedding_from_image(img_cv2):
    """
    Trích xuất vector đặc trưng khuôn mặt từ ảnh đã load (OpenCV format).
    """
    if img_cv2 is None:
        logger.error("Ảnh đầu vào cho extract_embedding_from_image là None.")
        raise ValueError("Ảnh đầu vào không hợp lệ (None).")

    logger.info(f"🔍 Trích xuất đặc trưng với model: {MODEL_NAME}, detector: {DETECTOR_NAME}")
    try:
        embedding_objects = DeepFace.represent(
            img_path=img_cv2,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_NAME,
            enforce_detection=True, # Bắt buộc tìm thấy khuôn mặt trong ảnh thật
            align=True
        )

        if not embedding_objects or not isinstance(embedding_objects, list) or not embedding_objects[0]:
            # Lỗi này có thể không bao giờ xảy ra nếu enforce_detection=True vì nó sẽ raise ValueError trước
            logger.error("Không trích xuất được đối tượng embedding hợp lệ.")
            raise ValueError("Không trích xuất được đặc trưng khuôn mặt từ ảnh.")

        first_face_obj = embedding_objects[0]
        embedding_vector = first_face_obj['embedding']
        facial_area = first_face_obj['facial_area']
        face_confidence = first_face_obj.get('face_confidence', None)

        return embedding_vector, facial_area, face_confidence
    except ValueError as ve: # Bắt lỗi cụ thể từ DeepFace.represent khi không tìm thấy mặt
        logger.error(f"Lỗi từ DeepFace.represent: {ve}")
        raise ValueError(f"Không thể phát hiện hoặc xử lý khuôn mặt trong ảnh: {ve}")
    except Exception as e:
        logger.exception(f"Lỗi không xác định khi trích xuất embedding: {e}")
        raise RuntimeError(f"Lỗi không xác định trong quá trình trích xuất embedding: {e}")