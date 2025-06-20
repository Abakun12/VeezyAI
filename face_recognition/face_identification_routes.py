import os
import config
from flask import request, jsonify, current_app
from deepface import DeepFace
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
MONGO_CONN_STR  = os.environ.get("MONGO_CONN_STR", "mongodb+srv://thuanchce170133:ZEQ16jqwjtolxbaV@cluster0.ajszsll.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
MONGO_DB_NAME = os.environ.get("MONGO_DB_NAME", "veezy_db")
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
        projection = {"_id": 1, "username": 1, "embedding": 1}
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
        threshold_from_verify = None

        for person_data in known_people_from_db:
            stored_embedding_vector = person_data.get("embedding")
            current_user_id = person_data.get("_id")

            if not stored_embedding_vector or not isinstance(stored_embedding_vector, list):
                logger.warning(f"Bỏ qua _id: {current_user_id} do thiếu embedding hoặc embedding không hợp lệ.")
                continue

            try:
                # Sử dụng DeepFace.verify để so sánh embedding của ảnh live với từng stored_embedding
                # DeepFace.verify sẽ tự trích xuất lại embedding từ live_img_cv2 nếu cần,
                # hoặc có thể tối ưu hơn nếu truyền trực tiếp live_embedding_vector (cần xem tài liệu DeepFace.verify)
                # Để đảm bảo, ta truyền img1_path là ảnh live, img2_path là embedding đã lưu.
                verification_result = DeepFace.verify(
                    img1_path=live_img_cv2,  # Ảnh live
                    img2_path=np.array(stored_embedding_vector, dtype=np.float32),
                    # Embedding từ DB, cần là numpy array
                    model_name=MODEL_NAME,
                    detector_backend=DETECTOR_NAME,  # Nên dùng để DeepFace tự xử lý việc phát hiện trên img1_path
                    distance_metric=DISTANCE_METRIC,
                    align=True,  # DeepFace sẽ tự căn chỉnh img1_path nếu cần
                    enforce_detection=True  # Đảm bảo khuôn mặt được tìm thấy trong img1_path để so sánh
                )

                current_distance = verification_result["distance"]
                threshold_from_verify = verification_result["threshold"]  # Lấy ngưỡng model đã dùng

                # logger.debug(f"So sánh với {person_data.get('username', current_user_id)}: verified={verification_result['verified']}, distance={current_distance:.4f}, threshold={threshold_from_verify}")

                if verification_result["verified"] and current_distance < min_distance_found:
                    min_distance_found = current_distance
                    best_match_person_info = person_data

            except ValueError as ve_verify:  # Lỗi từ DeepFace.verify
                logger.debug(f"Lỗi verify khi so sánh với {person_data.get('username', current_user_id)}: {ve_verify}")
                # Nếu lỗi là không tìm thấy mặt trong live_img_cv2, nó sẽ raise ở lần verify đầu tiên
                # và được bắt bởi try-except bên ngoài.
            except Exception as e_verify:
                logger.error(
                    f"Lỗi không xác định khi verify với {person_data.get('username', current_user_id)}: {e_verify}")

        if best_match_person_info:
            user_id_found = best_match_person_info.get("_id")
            user_name_found = best_match_person_info.get("username", user_id_found)
            logger.info(
                f"Kết quả nhận diện 1:N (DB): Tìm thấy {user_name_found} (ID: {user_id_found}) với khoảng cách {min_distance_found:.4f}")
            return jsonify({
                "message": "Nhận diện khuôn mặt thành công. Người dùng được xác định.",
                "identified_person": {
                    "_id": user_id_found,
                    "username": user_name_found,
                    "distance": round(min_distance_found, 4),
                    "threshold_used": threshold_from_verify  # Trả về ngưỡng mà DeepFace đã dùng
                },
                "login_successful": True
            }), 200
        else:
            logger.info("Kết quả nhận diện 1:N (DB): Không tìm thấy người nào khớp.")
            return jsonify({
                "message": "Nhận diện khuôn mặt không thành công. Không tìm thấy người dùng phù hợp.",
                "identified_person": None,
                "login_successful": False,
                "threshold_that_would_be_used": threshold_from_verify if threshold_from_verify is not None else DeepFace.verification.get_threshold(
                    MODEL_NAME, DISTANCE_METRIC)  # Cung cấp ngưỡng tham khảo
            }), 200

    except ValueError as ve_outer:  # Lỗi chung khi xử lý ảnh live hoặc dữ liệu đầu vào
        logger.error(f"Lỗi xử lý yêu cầu nhận diện (ValueError): {str(ve_outer)}")
        if "Face could not be detected" in str(ve_outer) or "multiple faces" in str(ve_outer):
            return jsonify({
                               "error": f"Không thể phát hiện khuôn mặt trong ảnh trực tiếp hoặc phát hiện nhiều hơn một khuôn mặt. (Live image face detection issue: {str(ve_outer)})"}), 400
        return jsonify({"error": f"Lỗi dữ liệu đầu vào khi nhận diện: {str(ve_outer)}"}), 400
    except Exception as e_outer:
        logger.exception(f"Lỗi không xác định trong quá trình nhận diện: {str(e_outer)}")
        return jsonify({"error": f"Đã xảy ra lỗi nội bộ khi nhận diện: {str(e_outer)}"}), 500

