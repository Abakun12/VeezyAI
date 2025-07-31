from deepface import DeepFace
from deepface.modules import verification
import numpy as np
import cv2
import logging
from flask import request, jsonify
from pymongo import MongoClient
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = config.MODEL_NAME
DETECTOR_NAME = config.DETECTOR_NAME

try:
    mongo_client = MongoClient(config.MONGO_CONN_STR, serverSelectionTimeoutMS=5000)
    mongo_client.admin.command('ping')
    db = mongo_client[config.MONGO_DB_NAME]
    face_logs_collection = db[config.FACE_LOGS_COLLECTION]
    account_face_collection = db[config.ACCOUNTS_COLLECTION]
    logger.info(f"Successfully connected to MongoDB.")
except Exception as e:
    logger.exception(f"INITIAL MONGODB CONNECTION FAILED: {e}. The service might not function correctly.")
    mongo_client = None
    face_logs_collection = None


def enroll_or_update_face():
    """
    Handles both new face enrollment and updating an existing face.
    Prevents a face from being registered to more than one account.
    """
    if account_face_collection is None:
        return jsonify({"error": "System error: Unable to connect to database."}), 500

    if request.method != 'POST':
        return jsonify({"error": "Method not allowed. Please use POST."}), 405

    # accountId is optional, only sent when a user wants to UPDATE their face
    account_id_to_update = request.form.get('accountId')

    if 'file' not in request.files:
        return jsonify({"error": "The 'file' field is not present in the request."}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No files selected."}), 400

    try:
        images_stream = file.read()
        np_image = np.frombuffer(images_stream, np.uint8)
        img_cv2 = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

        if img_cv2 is None:
            return jsonify({"error": "Không thể đọc được file ảnh."}), 400

        logger.info(f"Image read successfully: {file.filename}")

        # Extract embedding from the new face
        embedding_objects = DeepFace.represent(
            img_path=img_cv2,
            model_name=MODEL_NAME, detector_backend=DETECTOR_NAME,
            enforce_detection=True, anti_spoofing=True, align=True
        )
        new_embedding = embedding_objects[0]['embedding']

        # 1. Base query: find all accounts that already have an embedding
        query = {"faceEmbedding": {"$exists": True, "$ne": None}}

        # 2. If this is an UPDATE, exclude the current user's account from the check
        if account_id_to_update:
            logger.info(
                f"Performing an UPDATE for AccountId: {account_id_to_update}. Excluding this account from duplicate check.")
            # Add a "not equal" condition to the query
            query["_id"] = {"$ne": account_id_to_update}
        else:
            logger.info("Performing a NEW enrollment. Checking against all existing accounts.")

        # 3. Execute the query
        existing_accounts_to_check = list(account_face_collection.find(query))

        # Use a strict threshold for enrollment to ensure uniqueness
        ENROLL_THRESHOLD = 0.30

        # 4. Loop and compare
        for account in existing_accounts_to_check:
            stored_embedding = account.get("faceEmbedding")
            if stored_embedding:
                distance = verification.find_distance(new_embedding, stored_embedding, config.DISTANCE_METRIC)

                logger.info(f"Comparing with AccountId: {account.get('_id')}. Calculated Distance: {distance:.4f}")
                if distance < ENROLL_THRESHOLD:
                    logger.warning(
                        f"Enrollment failed. Face is too similar to existing AccountId: {account.get('AccountId')}. Distance: {distance:.4f}")
                    return jsonify({"error": "This face is already registered to another account."}), 409

        # If no duplicates are found, the face is valid
        response_data = {
            "message": "Valid face, can be registered or updated.",
            "embedding": new_embedding
        }
        return jsonify(response_data), 200

    except ValueError as ve:
        error_message = str(ve).lower()
        log_id = account_id_to_update or "N/A"
        if 'spoof' in error_message:
            logger.warning(f"Spoof attempt detected for AccountId: {log_id}")
            return jsonify({"error": "Fake detected. Please use live photo."}), 403
        else:
            logger.warning(f"Face detection failed for AccountId: {log_id}. Error: {ve}")
            return jsonify({"error": "No face detected in photo."}), 400
    except Exception as e:
        log_id = account_id_to_update or "N/A"
        logger.exception(f"An unexpected error occurred during face enrollment for AccountId: {log_id}. Error: {e}")
        return jsonify({"error": "A system error occurred."}), 500


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

def buy_ticket_by_face():
    """
    Nhận eventId + ảnh khuôn mặt, trả về embedding nếu chưa từng mua vé cho event này.
    Nếu đã có embedding khớp trong FaceLogs thì trả về lỗi.
    """
    if 'file' not in request.files or 'eventId' not in request.form:
        return jsonify({"error": "Missing photos or events."}), 400

    event_id = request.form['eventId']
    file = request.files['file']

    logger.info(f"Received request to buy ticket by face for EventId: {event_id}")

    if file.filename == '':
        return jsonify({"error": "No image file."}), 400

    try:
        np_image = np.frombuffer(file.read(), np.uint8)
        img_cv2 = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
        if img_cv2 is None:
            return jsonify({"error": "Cannot read image"}), 400
        logger.info(f"Image read successfully for EventId: {event_id}")
        embedding_objs = DeepFace.represent(
            img_path=img_cv2,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_NAME,
            enforce_detection=True,
            anti_spoofing=True,
            align=True
        )
        if not embedding_objs or not isinstance(embedding_objs, list) or not embedding_objs[0]:
            return jsonify({"error": "Do not extract facial features from images."}), 400

        embedding_vector = embedding_objs[0]['embedding']
        logger.info(f"Successfully extracted new face embedding for EventId: {event_id}")
        # Try vấn FaceLogs với eventId, so sánh embedding

        LOGIN_THRESHOLD_DISTANCE = 0.30

        pipeline = [
            {
                "$search": {
                    "index": "your_face_logs_index_name",
                    "knnBeta": {
                        "vector": embedding_vector,
                        "path": "faceEmbedding",
                        "k": 1,
                        "filter": {
                            "equals": {"path": "eventId", "value": event_id}
                        }
                    }
                }
            },
            {
                # Lấy cả embedding đã lưu để so sánh lại
                "$project": {"faceEmbedding": 1, "score": {"$meta": "searchScore"}}
            }
        ]
        logger.info(f"Executing Atlas Search pipeline for EventId {event_id}: {pipeline}")
        existing_logs = list(face_logs_collection.aggregate(pipeline))
        logger.info(f"Found {len(existing_logs)} potential matches from Atlas Search.")
        # Nếu Atlas Search tìm thấy một ứng viên
        if existing_logs:
            # Lấy embedding của ứng viên đó
            stored_embedding = existing_logs[0].get("faceEmbedding")
            search_score = existing_logs[0].get("score")

            # === LOGGING: Ghi lại điểm số từ Atlas Search ===
            logger.info(f"Candidate found with search score: {search_score:.4f}")
            # Dùng chính hàm của DeepFace để tính lại khoảng cách -> Đảm bảo nhất quán
            distance = verification.find_distance(embedding_vector, stored_embedding, config.DISTANCE_METRIC)

            # So sánh khoảng cách tính được với ngưỡng
            logger.info(f"Recalculated distance with DeepFace: {distance:.4f}")
            if distance < LOGIN_THRESHOLD_DISTANCE:
                logger.warning(
                    f"Attempt to buy ticket with an existing face for EventId {event_id}. "
                    f"Distance: {distance:.4f}"
                )
                return jsonify({"error": "Tickets are already in FaceLogs for this eventId."}), 400
            else:
                # === LOGGING: Ghi lại khi tìm thấy nhưng không đủ gần ===
                logger.info(f"Candidate found but distance {distance:.4f} is above threshold. Allowing purchase.")
        else:
            logger.info("No similar faces found in logs for this event. Allowing purchase.")
        # Nếu không có ứng viên nào hoặc ứng viên không đủ gần, cho phép mua vé
        return jsonify({
            "message": "Tickets can be purchased with this face.",
            "embedding": embedding_vector
        }), 200


    except ValueError as ve:
        error_message = str(ve)

        if "face could not be detected" in error_message:
            return jsonify({"error": "No face detected in photo."}), 400

        if "is fake" in error_message:
            return jsonify({"error": "Fake behavior detected. Please use live photo."}), 400

        return jsonify({"error": f"Face processing error: {error_message}"}), 400

    except Exception as e:
        logger.exception(f"Lỗi không xác định: {str(e)}")

        return jsonify({"error": f"Unknown system error: {str(e)}"}), 500
