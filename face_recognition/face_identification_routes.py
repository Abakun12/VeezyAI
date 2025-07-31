import config
from flask import request, jsonify
from deepface import DeepFace
from deepface.modules import verification
from pymongo import MongoClient
import numpy as np
import cv2
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = config.MODEL_NAME
DETECTOR_NAME = config.DETECTOR_NAME
DISTANCE_METRIC = config.DISTANCE_METRIC

# --- Cấu hình Kết nối MongoDB ---
MONGO_CONN_STR  = config.MONGO_CONN_STR
MONGO_DB_NAME = config.MONGO_DB_NAME

mongo_client = None
embeddings_collection = None

try:
    mongo_client = MongoClient(MONGO_CONN_STR, serverSelectionTimeoutMS=5000)
    mongo_client.admin.command('ping')
    db = mongo_client[MONGO_DB_NAME]
    embeddings_collection = db[config.ACCOUNTS_COLLECTION]
    face_logs_collection = db[config.FACE_LOGS_COLLECTION]
    logger.info(f"Đã kết nối thành công tới MongoDB: DB '{MONGO_DB_NAME}', Collection '{config.ACCOUNTS_COLLECTION}, {config.FACE_LOGS_COLLECTION}'")
except Exception as e:
    logger.exception(f"LỖI KẾT NỐI MONGODB ban đầu: {e}. Service sẽ không thể thực hiện nhận diện dựa trên DB.")

def _initialize_models():
    """Hàm này được gọi một lần khi server khởi động để làm nóng model."""
    try:
        logger.info("Bắt đầu làm nóng các model AI...")
        # Tạo một ảnh đen nhỏ để thực hiện một lệnh represent giả
        dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
        DeepFace.represent(dummy_image, model_name=config.MODEL_NAME, enforce_detection=False)
        logger.info("Các model AI đã được tải vào bộ nhớ (warm-up thành công).")
    except Exception as e:
        logger.error(f"Lỗi khi làm nóng model: {e}")

def _find_best_match(live_embedding_vector, known_people_from_db, threshold):
    """
    Tìm người dùng hiện tại có khoảng cách nhất giữa live_embedding_vector và
    các embedding của các người dùng trong DB.
    """
    best_match_person_info = None
    min_distance_found = float('inf')

    for person_data in known_people_from_db:
        stored_embedding_vector = person_data.get("faceEmbedding")
        if not stored_embedding_vector: continue

        distance = verification.find_distance(live_embedding_vector, stored_embedding_vector, DISTANCE_METRIC)
        if distance < min_distance_found and distance <= threshold:
            min_distance_found = distance
            best_match_person_info = person_data
        elif distance < min_distance_found:
            min_distance_found = distance
    return best_match_person_info, min_distance_found

# HÀM MỚI DÀNH RIÊNG CHO VIỆC LOGIN
def identify_face_from_image_and_db():
    """
    Nhận diện khuôn mặt 1:N (Tối ưu hiệu năng).
    """
    if embeddings_collection is None:
        return jsonify({"error": "Server configuration error: Unable to connect to database."}), 500

    if 'file' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    live_image_file = request.files['file']
    if live_image_file.filename == '':
        return jsonify({"error": "No selected live image file"}), 400

    try:
        # --- BƯỚC 1: Xử lý ảnh live và trích xuất embedding MỘT LẦN DUY NHẤT ---
        np_live_login = np.frombuffer(live_image_file.read(), np.uint8)
        live_img_cv2 = cv2.imdecode(np_live_login, cv2.IMREAD_COLOR)

        if live_img_cv2 is None:
            return jsonify({"error": "Could not decode live image"}), 400

        logger.info("Bắt đầu kiểm tra toàn vẹn ảnh: Số lượng và Liveness...")

        face_objs = DeepFace.extract_faces(
            img_path=live_img_cv2,
            detector_backend=config.DETECTOR_NAME,
            enforce_detection=True,
            anti_spoofing=True
        )

        if len(face_objs) != 1:
            logger.warning(f"Ảnh không hợp lệ. Phát hiện {len(face_objs)} khuôn mặt.")
            return jsonify({'error': "Please make sure there is only one face in the photo."}), 400

        if not face_objs[0]['is_real']:
            logger.warning("Cảnh báo giả mạo: Khuôn mặt có thể không phải người thật.")
            return jsonify({"error": "Suspicious behavior detected. Please use live photos."}), 403

        validated_face_obj = face_objs[0]

        # Gọi hàm trích xuất embedding từ ảnh live
        live_embedding_vector = DeepFace.represent(
            img_path=validated_face_obj['face'],
            model_name=config.MODEL_NAME,
            enforce_detection=False,
            detector_backend='skip'
        )[0]['embedding']
        logger.info("Đã trích xuất embedding từ ảnh live thành công.")

        # --- BƯỚC 2: Lấy dữ liệu từ MongoDB ---
        known_people_from_db = list(embeddings_collection.find({}, {"_id": 1, "username": 1, "faceEmbedding": 1}))
        if not known_people_from_db:
            return jsonify({"message": "There is no user data in the DB.", "identified_person": None}), 200

        # --- BƯỚC 3: So sánh hiệu quả và tìm người khớp nhất ---
        threshold = verification.find_threshold(config.MODEL_NAME, config.DISTANCE_METRIC)
        LOGIN_THRESHOLD = 0.30
        best_match_person, min_distance_found = _find_best_match(live_embedding_vector, known_people_from_db, LOGIN_THRESHOLD)

        # --- BƯỚC 4: Trả về kết quả ---
        if best_match_person:
            user_id_found = str(best_match_person.get("_id"))
            user_name_found = best_match_person.get("username", user_id_found)
            final_distance = min_distance_found
            logger.info(f"Kết quả: Tìm thấy {user_name_found} với khoảng cách {final_distance:.4f}")
            return jsonify({
                "message": "Identification successful. Login authorized.",
                "identified_person": {
                    "_id": user_id_found,
                    "username": user_name_found,
                    "distance": round(final_distance, 4),
                },
                "login_successful": True,
                "threshold_used": threshold
            }), 200
        else:
            logger.info("Kết quả: Không tìm thấy người nào khớp.")
            return jsonify({
                "message": "Identification failed. No matching user found.",
                "identified_person": None,
                "login_successful": False,
                "closest_match_distance": round(min_distance_found, 4) if min_distance_found != float('inf') else None,
                "threshold_that_was_used": threshold
            }), 200

    except ValueError as ve:
        logger.error(f"Lỗi xử lý khuôn mặt (ValueError): {str(ve)}")
        return jsonify({"error": f"Cannot detect faces in live photos: {str(ve)}"}), 400
    except Exception as eb:
        logger.exception(f"Lỗi không xác định: {str(eb)}")
        return jsonify({"error": f"An internal error occurred: {str(eb)}"}), 500

def identify_face_for_check_in():
    """
    Nhận eventId + ảnh khuôn mặt, truy vấn FaceLogs, nếu tìm thấy embedding khớp thì trả về orderId.
    Nếu không tìm thấy thì trả về lỗi.
    """
    if face_logs_collection is None:
        return jsonify({"error": "Lỗi hệ thống: Không thể kết nối tới cơ sở dữ liệu."}), 500

    if 'file' not in request.files or 'eventId' not in request.form:
        return jsonify({"error": "Thiếu file ảnh hoặc eventId."}), 400

    event_id = request.form['eventId']
    live_image_file = request.files['file']

    if live_image_file.filename == '':
        return jsonify({"error": "Không có file ảnh."}), 400

    try:
        np_live_image = np.frombuffer(live_image_file.read(), np.uint8)
        live_img_cv2 = cv2.imdecode(np_live_image, cv2.IMREAD_COLOR)
        if live_img_cv2 is None:
            return jsonify({"error": "Unable to read image"}), 400

        embedding_objs = DeepFace.represent(
            img_path=live_img_cv2,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_NAME,
            enforce_detection=True,
            anti_spoofing=True,
            align=True
        )
        if not embedding_objs or not isinstance(embedding_objs, list) or not embedding_objs[0]:
            return jsonify({"error": "Do not extract facial features from images."}), 400

        live_embedding_vector = embedding_objs[0]['embedding']

        CHECKIN_THRESHOLD_DISTANCE = 0.30

        # 2. Dùng Atlas Search để tìm ứng viên gần nhất
        pipeline = [
            {
                "$search": {
                    "index": "your_face_logs_index_name",
                    "knnBeta": {
                        "vector": live_embedding_vector,
                        "path": "faceEmbedding",
                        "k": 1,
                        "filter": {
                            "equals": {"path": "eventId", "value": event_id}
                        }
                    }
                }
            },
            {
                "$project": {
                    "orderId": 1,
                    "faceEmbedding": 1
                }
            }
        ]

        best_match_results = list(face_logs_collection.aggregate(pipeline))

        # 3. Xác thực ứng viên tìm được bằng DeepFace
        if best_match_results:
            best_match_log = best_match_results[0]
            stored_embedding = best_match_log.get("faceEmbedding")

            distance = verification.find_distance(live_embedding_vector, stored_embedding, DISTANCE_METRIC)

            # So sánh khoảng cách thực tế với ngưỡng CỐ ĐỊNH
            if distance < CHECKIN_THRESHOLD_DISTANCE:
                order_id = best_match_log.get("orderId")
                logger.info(f"Check-in thành công cho OrderId: {order_id} với khoảng cách {distance:.4f}")
                return jsonify({
                    "message": "Check-in successful.",
                    "orderId": order_id,
                    "distance": round(distance, 4)
                }), 200

        # Nếu không có ứng viên hoặc ứng viên không đủ gần
        logger.warning(f"Check-in thất bại. Ngưỡng khoảng cách yêu cầu: < {CHECKIN_THRESHOLD_DISTANCE}")
        return jsonify({"error": "Ticket purchased for this face not found."}), 404

    except ValueError as ve:
        error_message = str(ve)
        if "face could not be detected" in error_message:
            return jsonify({"error": "Faces cannot be detected in the photo."}), 400
        if "is fake" in error_message:
            return jsonify({"error": "Phát hiện hành vi giả mạo. Vui lòng sử dụng ảnh trực tiếp."}), 400
        return jsonify({"error": f"Face processing error: {error_message}"}), 400
    except Exception as e:
        logger.exception(f"Lỗi không xác định trong quá trình check-in: {str(e)}")
        return jsonify({"error": f"Unknown system error: {str(e)}"}), 500


