from flask import Flask, request, jsonify
import pandas as pd
import logging
import config
import data_loader
from recommendation.hybrid import get_recommendations_for_user
from face_recognition.face_enroll import enroll_face, preload_models
from face_recognition.verify_face import verify_face
from face_recognition.face_identification_routes import identify_face_from_image_and_db
from recommendation.content_based import build_event_profiles
from recommendation.collaborative import train_cf_model_sklearn
app = Flask(__name__)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- CÁC BIẾN TOÀN CỤC ĐỂ LƯU MODEL VÀ DỮ LIỆU ---
USERS_DF = None
EVENTS_DF = None
FEEDBACK_DF = None
EVENT_PROFILES = None
CF_MODEL_INFO = None

def initialize_and_train():
    """
    Hàm này được gọi một lần duy nhất khi server khởi động.
    """
    global USERS_DF, EVENTS_DF, FEEDBACK_DF, EVENT_PROFILES, CF_MODEL_INFO
    logger.info("Đang tạo model CF và tạo dữ liệu mẫu...")
    try:
        USERS_DF, EVENTS_DF, FEEDBACK_DF = data_loader.load_data_from_mongodb()
        EVENT_PROFILES, _= build_event_profiles(EVENTS_DF)
        CF_MODEL_INFO = train_cf_model_sklearn(USERS_DF)
        logger.info("--- KẾT THÚC PHA OFFLINE. SERVER SẴN SÀNG NHẬN REQUEST. ---")
    except Exception as e:
        logger.exception(f"Lỗi khi tạo model CF và dữ liệu mẫu: {e}")

@app.route('/')
def index():
    return "Recommendation Service for Veezy AI is running."

@app.route('/recommend', methods=['POST'])
def recommend_for_user_api():
    """
    API endpoint để Backend gọi.
    Mong đợi một JSON body chứa "user_id"
    Ví dụ: { "user_id": "user_A"}
    """
    try:
        request_data = request.get_json()
        if not request_data:
            return jsonify({"error": "Request body phải là JSON."}), 400

        user_id = request_data.get('user_id')
        top_k = 10

        if not user_id:
            return jsonify({"error": "Thiếu 'user_id' trong JSON body."}), 400

    except Exception as e:
        logger.error(f"Lỗi khi xử lý request JSON: {e}")
        return jsonify({"error": "Request JSON không hợp lệ."}), 400

    logger.info(f"Nhận được yêu cầu gợi ý cho người dùng: {user_id} với top_k={top_k}")

    if EVENT_PROFILES is None or CF_MODEL_INFO is None:
        logger.error("Các mô hình gợi ý chưa được huấn luyện hoặc có lỗi khi khởi động.")
        return jsonify({"error": "Service is not ready, please try again later."}), 503

    if USERS_DF is not None and user_id not in USERS_DF['AccountId'].values:
        return jsonify({"error": f"Không tìm thấy người dùng với ID: {user_id}"}), 404

    try:
        recommendations_ids =  get_recommendations_for_user(user_id, USERS_DF, EVENTS_DF, FEEDBACK_DF, EVENT_PROFILES, CF_MODEL_INFO, top_k=top_k)

        if not recommendations_ids:
            return jsonify({ "user_id": user_id, "recommendations": [] })

        recommended_events_details = EVENTS_DF[EVENTS_DF['eventId'].isin(recommendations_ids)].to_dict(orient='records')

        ordered_results = sorted(recommended_events_details, key=lambda x: recommendations_ids.index(x['eventId']))

        return jsonify({
            "user_id": user_id,
            "recommendations": ordered_results
        })
    except Exception as e:
        logger.exception(f"Lỗi khi tạo gợi ý cho người dùng {user_id}: {e}")
        return jsonify({"error": "Đã xảy ra lỗi nội bộ trong quá trình tạo gợi ý."}), 500

@app.route('/ai/enroll', methods=['POST'])
def enroll_face_endpoint():
    return enroll_face()

@app.route('/ai/verify', methods=['POST'])
def verify_face_endpoint():
    return verify_face()

@app.route('/ai/identify',methods=['POST'])
def identify_face_endpoint():
    return identify_face_from_image_and_db()

if __name__ == '__main__':
    initialize_and_train()
    preload_models()
    app.run(host='0.0.0.0', port=5001, debug=True)
