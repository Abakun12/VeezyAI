import os

from flask import request, jsonify
from deepface import DeepFace
from pymongo import MongoClient
import numpy as np
import cv2
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = "Facenet"
DETECTOR_NAME = "retinaface"
DISTANCE_METRIC = "cosine"

try:
    THRESHOLD = DeepFace.verification.get_threshold(MODEL_NAME, DISTANCE_METRIC)
    logger.info(f"Ngưỡng xác định khi so sánh: {THRESHOLD}")
except Exception as e:
    logger.error(f"Lỗi khi tìm ngưỡng xác định khi so sánh: {e}")
    THRESHOLD = 0.4

# --- Cấu hình Kết nối MongoDB ---
MONGO_CONN_STR  = os.environ.get("MONGO_CONN_STR", "mongodb+srv://thuanchce170133:3OaqJRN0UWj2WI0V@cluster0.ajszsll.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
MONGO_DB_NAME = os.environ.get("MONGO_DB_NAME", "veezy_db")
MONGO_COLLECTION_NAME = os.environ.get("MONGO_COLLECTION_NAME", "")

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


def calculate_distance(embedding1, embedding2, metric="cosine"):
    """
    Hàm tính khoảng cách giữa 2 vector embedding.
    """
    emb1 = np.array(embedding1, dtype=np.float32).flatten()
    emb2 = np.array(embedding2, dtype=np.float32).flatten()

    if emb1.shape[0] == 0 or emb2.shape[0] == 0 or emb1.shape != emb2.shape:
        return float("inf")

    if metric == "cosine":
        dot = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        if norm1 == 0 or norm2 == 0:
            return float("inf")
        cos_sim = dot / (norm1 * norm2)

        return 1 - cos_sim
    elif metric == "euclidean":
        return np.linalg.norm(emb1 - emb2)
    elif metric == "euclidean_l2":
        if np.linalg.norm(emb1) == 0 or np.linalg.norm(emb2) == 0:
            return float("inf")
        emb1_normalized = emb1 / np.linalg.norm(emb1)
        emb2_normalized = emb2 / np.linalg.norm(emb2)
        return np.linalg.norm(emb1_normalized - emb2_normalized)
    else:
        logger.error(f"Hàm tính khoảng cách giữa 2 vector embedding không hợp lệ (Invalid metric: {metric})")
        raise ValueError("Invalid metric")


def identify_face_from_image_and_db():
    """
    Nhận diện khuôn mặt 1:N. Nhận file ảnh, AI tự lấy danh sách embedding từ MongoDB.
    Có thể nhận 'event_id' (tùy chọn từ form data) để lọc phạm vi tìm kiếm trong DB.
    """
    global embeddings_collection

    if embeddings_collection is None:
        logger.error("Kết nối MongoDB không được thiết lập. Không thể truy vấn embeddings.")
        return jsonify({"error": "Lỗi cấu hình server: Không thể kết nối cơ sở dữ liệu."}), 500

    if 'file' not in request.files:
        logger.warning("Không có file ảnh nào được cung cấp trong request.")
        return jsonify({"error": "No image file provided"}), 400

    live_image_file = request.files['file']

    if live_image_file.filename == '':
        logger.warning("Tên file ảnh trực tiếp trống.")
        return jsonify({"error": "No selected live image file"}), 400

    try:
        live_image_stream = live_image_file.read()
        np_live_image = np.frombuffer(live_image_stream, np.uint8)
        live_img_cv2 = cv2.imdecode(np_live_image, cv2.IMREAD_COLOR)

        if live_img_cv2 is None:
            logger.error("Không thể giải mã hình ảnh trực tiếp.")
            return jsonify({"error": "Could not decode live image"}), 400
        logger.info(f"Đã nhận và giải mã ảnh: {live_image_file.filename} cho nhận diện 1:N (DB)")

        # 1. Trích xuất embedding từ ảnh
        try:
            live_embedding_objs = DeepFace.represent(
                img_path=live_img_cv2,
                model_name=MODEL_NAME,
                detector_backend=DETECTOR_NAME,
                enforce_detection=True,
                align=True
            )
            if not live_embedding_objs or not isinstance(live_embedding_objs, list) or not live_embedding_objs[0].get(
                    'embedding'):
                logger.warning("Không thể trích xuất embedding từ ảnh live.")
                return jsonify({"error": "Could not extract embedding from live image"}), 400
            live_embedding_vector = live_embedding_objs[0]['embedding']
            logger.info("Đã trích xuất embedding từ ảnh live thành công.")
        except ValueError as ve_rep:  # Lỗi không tìm thấy mặt trong ảnh
            logger.error(f"Lỗi khi trích xuất embedding cho ảnh live (1:N DB): {ve_rep}")
            return jsonify({"error": f"Không thể xử lý ảnh live cho tìm kiếm 1:N: {ve_rep}"}), 400
        except Exception as e_rep:  # Các lỗi khác khi represent
            logger.exception(f"Lỗi không xác định khi trích xuất embedding ảnh live: {e_rep}")
            return jsonify({"error": "Lỗi server khi xử lý ảnh live."}), 500

        # 2. Lấy embedding đã biết từ MongoDB

        query_filter = {}
        projection = {"user_id": 1, "name": 1, "embedding": 1, "_id": 0}
        try:
            known_people_from_db = list(embeddings_collection.find(query_filter, projection))
        except Exception as e_db:
            logger.exception(f"Lỗi khi truy vấn embeddings từ MongoDB: {e_db}")
            return jsonify({"error": "Lỗi server khi truy vấn cơ sở dữ liệu."}), 500

        if not known_people_from_db:
            logger.info("Không có dữ liệu embedding nào trong DB để so sánh.")
            return jsonify({"message": "Không có dữ liệu người dùng đã biết trong DB để nhận diện.",
                            "identified_person": None}), 200

        logger.info(f"Bắt đầu nhận diện 1:N với {len(known_people_from_db)} người từ DB...")

        best_match_person_info = None
        min_distance_found = float('inf')  # Khởi tạo với giá trị vô cùng lớn

        for person_data in known_people_from_db:
            stored_embedding_vector = person_data.get("embedding")

            if not stored_embedding_vector or not isinstance(stored_embedding_vector, list):
                logger.warning(
                    f"Bỏ qua user_id: {person_data.get('user_id')} do thiếu embedding hoặc embedding không hợp lệ.")
                continue

            try:
                # stored_embedding_vector đã là list các float từ MongoDB (nếu lưu đúng)
                distance = calculate_distance(live_embedding_vector, stored_embedding_vector, DISTANCE_METRIC)
                # logger.debug(f"So sánh với {person_data.get('name', person_data.get('user_id'))}: distance={distance:.4f}, threshold={THRESHOLD}")

                if distance < THRESHOLD and distance < min_distance_found:  # So sánh với ngưỡng và khoảng cách nhỏ nhất đã tìm thấy
                    min_distance_found = distance
                    best_match_person_info = person_data  # Lưu thông tin người khớp nhất

            except Exception as e_calc:  # Lỗi khi tính toán khoảng cách hoặc chuyển đổi embedding
                logger.error(
                    f"Lỗi khi tính toán khoảng cách hoặc xử lý embedding cho user_id {person_data.get('user_id')}: {e_calc}")

        if best_match_person_info:
            logger.info(
                f"Kết quả nhận diện 1:N (DB): Tìm thấy {best_match_person_info.get('name', best_match_person_info.get('user_id'))} với khoảng cách {min_distance_found:.4f}")
            return jsonify({
                "message": "Nhận diện 1:N (DB) hoàn tất. Tìm thấy người khớp.",
                "identified_person": {
                    "user_id": best_match_person_info.get("user_id"),
                    "name": best_match_person_info.get("name"),  # Trả về tên nếu có
                    "distance": round(min_distance_found, 4),  # Làm tròn khoảng cách
                    "threshold_used": THRESHOLD
                },
                "match_type": "one-to-many-db"
            }), 200
        else:
            logger.info("Kết quả nhận diện 1:N (DB): Không tìm thấy người nào khớp.")
            return jsonify({
                "message": "Nhận diện 1:N (DB) hoàn tất. Không tìm thấy người nào khớp.",
                "identified_person": None,
                "match_type": "one-to-many-db"
            }), 200

    except ValueError as ve_outer:
        logger.error(f"Lỗi xử lý yêu cầu nhận diện (ValueError): {str(ve_outer)}")
        if "Face could not be detected" in str(ve_outer) or "multiple faces" in str(ve_outer):
            return jsonify({
                               "error": f"Không thể phát hiện khuôn mặt trong ảnh trực tiếp hoặc phát hiện nhiều hơn một khuôn mặt. (Live image face detection issue: {str(ve_outer)})"}), 400
        return jsonify({"error": f"Lỗi dữ liệu đầu vào khi nhận diện: {str(ve_outer)}"}), 400
    except Exception as e_outer:
        logger.exception(f"Lỗi không xác định trong quá trình nhận diện: {str(e_outer)}")
        return jsonify({"error": f"Đã xảy ra lỗi nội bộ khi nhận diện: {str(e_outer)}"}), 500

