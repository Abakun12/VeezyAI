import os
import config
from flask import request, jsonify
from deepface import DeepFace
from deepface.modules import verification
from pymongo import MongoClient
import numpy as np
import cv2
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = config.MODEL_NAME
DETECTOR_NAME = config.DETECTOR_NAME
DISTANCE_METRIC = config.DISTANCE_METRIC


# --- Cấu hình Kết nối MongoDB ---
MONGO_CONN_STR  = config.MONGO_CONN_STR
MONGO_DB_NAME = config.MONGO_DB_NAME
MONGO_COLLECTION_NAME = os.environ.get("MONGO_COLLECTION_NAME", "Accounts")

mongo_client = None
embeddings_collection = None

try:
    mongo_client = MongoClient(MONGO_CONN_STR, serverSelectionTimeoutMS=5000)
    mongo_client.admin.command('ping')
    db = mongo_client[MONGO_DB_NAME]
    embeddings_collection = db[MONGO_COLLECTION_NAME]
    logger.info(f"Đã kết nối thành công tới MongoDB: DB '{MONGO_DB_NAME}', Collection '{MONGO_COLLECTION_NAME}'")
except Exception as e:
    logger.exception(f"LỖI KẾT NỐI MONGODB ban đầu: {e}. Service sẽ không thể thực hiện nhận diện dựa trên DB.")

def _find_best_match(live_embedding_vector, known_people_from_db, threshold):
    """
    Tìm người dùng hiện tại có khoảng cách nhất giữa live_embedding_vector và
    các embedding của các người dùng trong DB.
    """
    best_match_person_info = None
    min_distance_found = float('inf')

    for person_data in known_people_from_db:
        stored_embedding_vector = person_data.get("embedding")
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
        return jsonify({"error": "Lỗi cấu hình server: Không thể kết nối cơ sở dữ liệu."}), 500

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

        try:
            face_objs = DeepFace.extract_faces(
                img_path=live_img_cv2,
                detector_backend=DETECTOR_NAME,
                enforce_detection=True,
                anti_spoofing=True
            )

            if len(face_objs) != 1:
                logger.warning(f"Ảnh không hợp lệ. Phát hiện {len(face_objs)} khuôn mặt.")
                return jsonify({'error': "Vui lòng đảm bảo chỉ có một khuôn mặt trong ảnh."}), 400

            if not face_objs[0]['is_real']:
                logger.warning("Cảnh báo giả mạo: Khuôn mặt có thể không phải người thật.")
                return jsonify({"error": "Phát hiện hành vi đáng ngờ. Vui lòng sử dụng ảnh chụp trực tiếp."}), 403
        except ValueError as ve:
            logger.error(f"Lỗi xử lý khuôn mặt (ValueError): {str(ve)}")
            return jsonify({"error": f"Không thể phát hiện khuôn mặt trong ảnh live: {str(ve)}"}), 400

        # Gọi hàm trích xuất embedding từ ảnh live
        live_embedding_vector = DeepFace.represent(
            img_path=live_img_cv2,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_NAME,
            enforce_detection=True,
            align=True
        )[0]['embedding']
        logger.info("Đã trích xuất embedding từ ảnh live thành công.")

        # --- BƯỚC 2: Lấy dữ liệu từ MongoDB ---
        known_people_from_db = list(embeddings_collection.find({}, {"_id": 1, "username": 1, "embedding": 1}))
        if not known_people_from_db:
            return jsonify({"message": "Không có dữ liệu người dùng trong DB.", "identified_person": None}), 200

        # --- BƯỚC 3: So sánh hiệu quả và tìm người khớp nhất ---
        threshold = verification.find_threshold(MODEL_NAME, DISTANCE_METRIC)
        best_match_person, min_distance_found = _find_best_match(live_embedding_vector, known_people_from_db, threshold)

        # --- BƯỚC 4: Trả về kết quả ---
        if best_match_person:
            user_id_found = str(best_match_person.get("_id"))  # Chuyển ObjectId thành string
            user_name_found = best_match_person.get("username", user_id_found)
            #final_distance = verification.find_distance(live_embedding_vector, best_match_person.get("embedding"), DISTANCE_METRIC)
            final_distance = min_distance_found
            logger.info(f"Kết quả: Tìm thấy {user_name_found} với khoảng cách {final_distance:.4f}")
            return jsonify({
                "message": "Nhận diện thành công. Đăng nhập được cấp phép.",
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
                "message": "Nhận diện không thành công. Không tìm thấy người dùng phù hợp.",
                "identified_person": None,
                "login_successful": False,
                "closest_match_distance": round(min_distance_found, 4) if min_distance_found != float('inf') else None,  # Cung cấp thêm thông tin
                "threshold_that_was_used": threshold
            }), 200

    except ValueError as ve:
        logger.error(f"Lỗi xử lý khuôn mặt (ValueError): {str(ve)}")
        return jsonify({"error": f"Không thể phát hiện khuôn mặt trong ảnh live: {str(ve)}"}), 400
    except Exception as eb:
        logger.exception(f"Lỗi không xác định: {str(eb)}")
        return jsonify({"error": f"Đã xảy ra lỗi nội bộ: {str(eb)}"}), 500


# HÀM MỚI DÀNH RIÊNG CHO VIỆC CHECK-IN
def identify_face_for_check_in():
    """
    Nhận diện khuôn mặt 1:N cho chức năng Check-in.
    Hàm này không quét toàn bộ DB mà chỉ so sánh với một danh sách ứng viên được cung cấp.
    """
    # --- BƯỚC 1: Nhận dữ liệu từ request ---
    if 'file' not in request.files:
        return jsonify({"error": "Không có file ảnh nào được gửi."}), 400

    live_image_file = request.files['file']
    if live_image_file.filename == '':
        return jsonify({"error": "Không có file ảnh nào được chọn."}), 400

    # Nhận danh sách ứng viên từ form data (được gửi từ backend ASP.NET)
    candidates_json = request.form.get('candidates')
    if not candidates_json:
        return jsonify({"error": "Không có danh sách ứng viên (candidates) nào được cung cấp."}), 400

    try:
        # Chuyển chuỗi JSON thành một list các dictionary
        candidates_from_request = json.loads(candidates_json)
        if not isinstance(candidates_from_request, list):
            raise TypeError("Candidates phải là một danh sách (list).")
    except (json.JSONDecodeError, TypeError) as eb:
        logger.error(f"Lỗi phân tích JSON từ candidates: {eb}")
        return jsonify({"error": "Dữ liệu candidates không phải là JSON list hợp lệ."}), 400

    if not candidates_from_request:
        return jsonify({"message": "Danh sách ứng viên rỗng."}), 200

    try:
        # --- BƯỚC 2: Xử lý ảnh live và trích xuất embedding ---
        np_live_check_in = np.frombuffer(live_image_file.read(), np.uint8)
        live_img_cv2 = cv2.imdecode(np_live_check_in, cv2.IMREAD_COLOR)

        if live_img_cv2 is None:
            return jsonify({"error": "Could not decode live image"}), 400

        logger.info("Bắt đầu kiểm tra toàn vẹn ảnh: Số lượng và Liveness...")
        face_objs = DeepFace.extract_faces(
            img_path=live_img_cv2,
            detector_backend=DETECTOR_NAME,
            enforce_detection=True,
            anti_spoofing=True
        )

        if len(face_objs) != 1:
            return jsonify({'error': "Vui lòng đảm bảo chỉ có một khuôn mặt trong ảnh."}), 400
        if not face_objs[0]['is_real']:
            return jsonify({"error": "Phát hiện hành vi đáng ngờ. Vui lòng sử dụng ảnh chụp trực tiếp."}), 403

        live_embedding_vector = DeepFace.represent(
            img_path=live_img_cv2,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_NAME,
            enforce_detection=True,
            align=True
        )[0]['embedding']
        logger.info("Check-in: Đã trích xuất embedding từ ảnh live thành công.")

        # --- BƯỚC 3: So sánh với danh sách ứng viên và tìm người khớp nhất ---

        threshold = verification.find_threshold(MODEL_NAME, DISTANCE_METRIC)
        best_match_person, min_distance_found = _find_best_match(live_embedding_vector, candidates_from_request, threshold)

        # --- BƯỚC 4: Trả về kết quả Check-in ---
        if best_match_person:
            user_id_found = best_match_person.get("user_id", "Không rõ")
            final_distance = min_distance_found
            logger.info(f"Check-in: Tìm thấy {user_id_found} với khoảng cách {final_distance:.4f}")
            return jsonify({
                "message": "Check-in thành công.",
                "check_in_successful": True,
                "identified_person": {
                    "user_id": user_id_found,
                    "distance": round(final_distance, 4),
                },
                "threshold_used": threshold
            }), 200
        else:
            logger.info(f"Check-in: Không tìm thấy người nào khớp. Khoảng cách gần nhất: {min_distance_found:.4f}")
            return jsonify({
                "message": "Check-in không thành công. Không tìm thấy người dùng phù hợp trong danh sách.",
                "check_in_successful": False,
                "closest_match_distance": round(min_distance_found, 4) if min_distance_found != float('inf') else None,
                "threshold_that_was_used": threshold
            }), 200

    except ValueError as ve:
        logger.error(f"Lỗi xử lý khuôn mặt (ValueError): {str(ve)}")
        return jsonify({"error": f"Không thể phát hiện khuôn mặt trong ảnh live: {str(ve)}"}), 400
    except Exception as eb:
        logger.exception(f"Lỗi không xác định trong quá trình check-in: {str(eb)}")
        return jsonify({"error": f"Đã xảy ra lỗi nội bộ: {str(eb)}"}), 500