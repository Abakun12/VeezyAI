import config
from flask import request, jsonify
from deepface import DeepFace
from deepface.modules import verification
from pymongo import MongoClient
import numpy as np
import cv2
import logging
from face_recognition import embedding_service

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
    logging.info("Initializing Custom Embedding Service...")
    # Không cần truyền embedding_dim vì model đã có sẵn
    custom_embedding_service = embedding_service.CustomEmbeddingService(model_path=config.CUSTOM_MODEL_PATH)
    logging.info("Custom Embedding Service initialized successfully.")
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
    Nhận eventsId + ảnh khuôn mặt, truy vấn FaceLog, nếu tìm thấy embedding khớp thì trả về orderId.
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

        # 1. Dùng extract_faces để tìm tất cả khuôn mặt và thông tin của chúng
        all_faces = DeepFace.extract_faces(
            img_path=live_img_cv2,
            detector_backend=DETECTOR_NAME,
            enforce_detection=True,
            anti_spoofing=True,
            align=True
        )

        # Nếu không tìm thấy khuôn mặt nào
        if not all_faces:
            logger.warning("Không tìm thấy khuôn mặt nào trong ảnh.")
            return jsonify({"error": "No face detected in the photo."}), 400

        # 2. Tìm khuôn mặt có diện tích lớn nhất (người gần camera nhất)
        largest_face_obj = None
        max_area = 0
        for face_obj in all_faces:
            facial_area = face_obj['facial_area']
            area = facial_area['w'] * facial_area['h']
            if area > max_area:
                max_area = area
                largest_face_obj = face_obj

        logger.info(f"Phát hiện {len(all_faces)} khuôn mặt, đã chọn khuôn mặt lớn nhất để check-in.")

        # 3. Trích xuất embedding CHỈ TỪ khuôn mặt lớn nhất đã chọn
        # Dùng detector_backend='skip' để tăng tốc vì khuôn mặt đã được cắt và căn chỉnh
        live_embedding_vector = DeepFace.represent(
            img_path=largest_face_obj['face'],
            model_name=MODEL_NAME,
            enforce_detection=False,
            detector_backend='skip'
        )[0]['embedding']

        CHECKIN_THRESHOLD_DISTANCE = DeepFace.verification.find_threshold(config.MODEL_NAME, config.DISTANCE_METRIC)

        # 2. Dùng Atlas Search để tìm ứng viên gần nhất
        pipeline = [
            {
                '$vectorSearch': {
                    "index": "face_logs_vector_index",
                    "path": "faceEmbedding",
                    "queryVector": live_embedding_vector,
                    "numCandidates": 10,
                    "limit": 1,
                    "filter": {
                        "eventId": event_id,
                        "isCheckin": {"$ne": True}
                    }
                }
            },
            {
                "$project": {
                    "orderId": 1,
                    "faceEmbedding": 1,
                    "userId": 1,
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
        ]

        best_match_results = list(face_logs_collection.aggregate(pipeline))

        # 3. Xác thực ứng viên tìm được bằng DeepFace
        if best_match_results:
            # Kịch bản 1: CÓ tìm thấy ứng viên gần nhất trong DB
            best_match_log = best_match_results[0]
            stored_embedding = best_match_log.get("faceEmbedding")
            distance = verification.find_distance(live_embedding_vector, stored_embedding, DISTANCE_METRIC)

            logger.info(
                f"Candidate found. Distance: {distance:.4f}. Required threshold: < {CHECKIN_THRESHOLD_DISTANCE:.4f}")

            if distance < CHECKIN_THRESHOLD_DISTANCE:
                # Trường hợp thành công: Khuôn mặt khớp
                order_id = best_match_log.get("orderId")
                userId = best_match_log.get("userId")
                logger.info(f"Check-in SUCCESS for OrderId: {order_id} with distance {distance:.4f}")
                return jsonify({
                    "message": "Check-in successful.",
                    "orderId": order_id,
                    "userId": userId,
                    "distance": round(distance, 4)
                }), 200
            else:
                # Trường hợp thất bại 1: TÌM THẤY NHƯNG SAI NGƯỜI
                order_id = best_match_log.get("orderId")
                logger.warning(
                    f"Check-in FAILED for OrderId: {order_id}. "
                    f"Face did not match (distance {distance:.4f} is over threshold). "
                    f"Possible impersonation attempt."
                )
                return jsonify({"error": "Face does not match the registered ticket holder."}), 403  # 403 Forbidden
        else:
            # Trường hợp thất bại 2: KHÔNG TÌM THẤY BẤT KỲ AI TƯƠNG TỰ
            logger.warning(
                f"Check-in FAILED for eventId: {event_id}. "
                f"No potential candidates found in the database for this event."
            )
            return jsonify(
                {"error": "No ticket purchased with a similar face was found for this event."}), 404  # 404 Not Found


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


# File: app.py (Sửa lại hàm này)

def _find_best_match_customs(live_embedding, known_people, threshold, distance_metric):  # <-- Thêm distance_metric vào đây
    """
    So sánh một embedding với danh sách các embedding đã biết.
    """
    min_distance = float('inf')
    best_match_person = None

    for person in known_people:
        stored_embedding = person.get("faceEmbedding")
        if stored_embedding:
            # Sử dụng distance_metric được truyền vào
            distance = verification.find_distance(
                live_embedding,
                np.array(stored_embedding),
                distance_metric  # <-- Sử dụng nó ở đây
            )

            # Cập nhật khoảng cách nhỏ nhất tìm được
            if distance < min_distance:
                min_distance = distance

            # Nếu tìm thấy người khớp trong ngưỡng, gán và dừng sớm
            if distance < threshold:
                best_match_person = person
                min_distance = distance  # Cập nhật khoảng cách cuối cùng là khoảng cách của người khớp
                break

    return best_match_person, min_distance

def check_image_sharpness(image_np, threshold):
    """Kiểm tra xem ảnh có bị mờ hay không bằng phương sai của Laplacian."""
    if image_np is None or image_np.size == 0:
        return False, "Invalid image for sharpness check."
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    if variance < threshold:
        logging.warning(f"Image too blurry. Laplacian variance: {variance:.2f} < {threshold}")
        return False, "Image is too blurry. Please use a sharper photo."
    logging.info(f"Sharpness check passed. Variance: {variance:.2f}")
    return True, "Image is sharp enough."

def check_image_brightness(image_np, threshold):
    """Kiểm tra xem ảnh có quá tối hay không bằng độ sáng trung bình."""
    if image_np is None or image_np.size == 0:
        return False, "Invalid image for brightness check."
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    brightness = gray.mean()
    if brightness < threshold:
        logging.warning(f"Image too dark. Mean brightness: {brightness:.2f} < {threshold}")
        return False, "Image is too dark. Please use better lighting."
    logging.info(f"Brightness check passed. Mean brightness: {brightness:.2f}")
    return True, "Image is bright enough."

def identify_face_from_image_and_db_custom():
    """
    Nhận diện khuôn mặt 1:N bằng model tùy chỉnh và logic query thật.
    """
    if not custom_embedding_service or not embeddings_collection:
        return jsonify({"error": "System error: A required service is not available."}), 500

    if 'file' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    live_image_file = request.files['file']
    if live_image_file.filename == '':
        return jsonify({"error": "No selected live image file"}), 400

    try:
        # --- BƯỚC 1: XỬ LÝ ẢNH LIVE ---
        np_live_login = np.frombuffer(live_image_file.read(), np.uint8)
        live_img_cv2 = cv2.imdecode(np_live_login, cv2.IMREAD_COLOR)

        if live_img_cv2 is None:
            return jsonify({"error": "Could not decode live image"}), 400

        logging.info("Step 1: Detecting face, checking for liveness and validity...")
        face_objs = DeepFace.extract_faces(
            img_path=live_img_cv2,
            detector_backend=config.DETECTOR_NAME_1,
            enforce_detection=True,
            anti_spoofing=True
        )

        if len(face_objs) != 1:
            return jsonify({'error': "Please make sure there is only one face in the photo."}), 400

        face_crop_np = (face_objs[0]['face'] * 255).astype(np.uint8)

        is_sharp, sharp_message = check_image_sharpness(face_crop_np, config.MIN_BLUR_THRESHOLD)
        if not is_sharp:
            return jsonify({"error": sharp_message}), 400

        # 3.4: Kiểm tra độ sáng (Brightness)
        is_bright, bright_message = check_image_brightness(face_crop_np, config.MIN_BRIGHTNESS_THRESHOLD)
        if not is_bright:
            return jsonify({"error": bright_message}), 400

        # --- BƯỚC 2: TRÍCH XUẤT EMBEDDING BẰNG MODEL TÙY CHỈNH ---
        logging.info("Step 2: Extracting embedding using custom model...")
        live_embedding_vector = custom_embedding_service.get_embedding(face_crop_np)

        # <<< LOGIC QUERY DATABASE THẬT CỦA BẠN ĐÃ ĐƯỢC CẬP NHẬT TẠI ĐÂY >>>
        # ======================================================================
        # --- BƯỚC 3: LẤY DỮ LIỆU TỪ MONGODB VÀ SO SÁNH ---
        logging.info("Step 3: Fetching known faces from database and comparing...")

        # Lấy tất cả document, nhưng chỉ lấy các trường cần thiết để tối ưu
        known_people_from_db = list(embeddings_collection.find(
            {},
            {"_id": 1, "username": 1, "faceEmbedding": 1}
        ))
        # ======================================================================

        if not known_people_from_db:
            logging.info("No user data found in the database to compare.")
            return jsonify({"message": "There is no user data in the DB to compare with."}), 200

        # Gọi hàm so sánh với ngưỡng đăng nhập từ file config
        best_match_person, min_distance_found = _find_best_match_customs(
            live_embedding_vector,
            known_people_from_db,
            config.CUSTOM_LOGIN_THRESHOLD,
            config.DISTANCE_METRIC_1
        )

        # --- BƯỚC 4: TRẢ VỀ KẾT QUẢ ---
        if best_match_person:
            user_id_found = str(best_match_person.get("_id"))
            user_name_found = best_match_person.get("username", user_id_found)

            logging.info(f"Success! Found match: {user_name_found} with distance: {min_distance_found:.4f}")
            return jsonify({
                "message": "Identification successful. Login authorized.",
                "identified_person": {
                    "_id": user_id_found,
                    "username": user_name_found,
                    "distance": round(min_distance_found, 4),
                },
                "login_successful": True,
            }), 200

        else:
            logger.info("Kết quả: Không tìm thấy người nào khớp.")
            return jsonify({
                "message": "Identification failed. No matching user found.",
                "identified_person": None,
                "login_successful": False,
                "closest_match_distance": round(min_distance_found, 4) if min_distance_found != float('inf') else None,
                "threshold_that_was_used": config.CUSTOM_LOGIN_THRESHOLD
            }), 200

    except ValueError as ve:
        logging.error(f"Face processing error (ValueError): {str(ve)}")
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        logging.error(f"An unknown error occurred during identification: {str(e)}", exc_info=True)
        return jsonify({"error": "An internal server error occurred."}), 500
