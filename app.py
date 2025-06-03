from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import cv2
import logging
from sklearn.metrics.pairwise import cosine_similarity
from recommendation.hybrid import get_hybrid_recommendations
from face_recognition.face_enroll import enroll_face, preload_models
from face_recognition.verify_face import verify_face
from face_recognition.face_identification_routes import identify_face_from_image_and_db
app = Flask(__name__)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sample_data_for_recommendation = {
    "userId": "user_sample_123",
    "userPreferences": {
        "preferredCategories": ["workshop", "concert"],
        "preferredTags": ["python", "data science", "live music"]
    },
    "activeEvents": [
        {"eventId": "event_701", "name": "Python Conference 2025", "tags": ["python", "programming"],
         "category": "conference"},
        {"eventId": "event_702", "name": "Data Science Summit", "tags": ["data science", "ai"], "category": "seminar"},
        {"eventId": "event_703", "name": "Indie Music Fest", "tags": ["live music", "indie"], "category": "concert"},
        {"eventId": "event_704", "name": "Web Dev Workshop", "tags": ["web", "programming"], "category": "workshop"}
    ],
    "interactions": [
        {"userId": "user_sample_123", "eventId": "event_prev_001", "checkedIn": True}
    ]
}


@app.route('/recommend', methods=['POST'])
def recommend():
    data_input = None
    if request.method == 'POST':
        try:
            data_input = request.get_json()
            if data_input is None:
                logger.warning("Request POST nhưng không có JSON body hoặc Content-Type sai, sử dụng dữ liệu mẫu.")
                data_input = sample_data_for_recommendation
        except Exception as e:
            logger.error(f"Lỗi khi parse JSON từ request: {e}. Sử dụng dữ liệu mẫu.")
            data_input = sample_data_for_recommendation
    else:  # Mặc định cho GET hoặc nếu không phải POST
        logger.info("Request GET đến /recommend, sử dụng dữ liệu mẫu.")
        data_input = sample_data_for_recommendation

    user_id = data_input.get('userId')
    if not user_id:
        return jsonify({"error": "userId is required in the input data"}), 400

    user_preferences = data_input.get('userPreferences', {})
    active_events_list = data_input.get('activeEvents', [])  #
    interactions_list = data_input.get('interactions', [])

    # Chuyển interactions_list thành DataFrame nếu hàm get_hybrid_recommendations yêu cầu
    interactions_df = pd.DataFrame(interactions_list)
    # Tương tự, active_events_df nếu cần
    # active_events_df = pd.DataFrame(active_events_list)

    # Lấy top_k và alpha từ query parameters nếu có, với giá trị mặc định
    top_k_req = request.args.get('top_k', default=3, type=int)
    alpha_req = request.args.get('alpha', default=0.6, type=float)

    try:
        recommendations = get_hybrid_recommendations(
            user_id=user_id,
            active_events=active_events_list,
            interactions_df=interactions_df,
            user_preferences=user_preferences,
            top_k=top_k_req,
            alpha=alpha_req,
        )
        return jsonify({"userId": user_id, "recommendations": recommendations}), 200
    except Exception as e:
        logger.exception(f"Lỗi trong quá trình tạo gợi ý cho user {user_id}: {e}")
        return jsonify({"error": "Lỗi xảy ra trong quá trình tạo gợi ý."}), 500


@app.route('/ai/enroll', methods=['POST'])
def enroll_face_endpoint():
    return enroll_face()

@app.route('/ai/verify', methods=['POST'])
def verify_face_endpoint():
    return verify_face()

@app.route('/ai/identify', methods=['POST'])
def identify_face_endpoint():
    return identify_face_from_image_and_db()

if __name__ == '__main__':
    preload_models()
    app.run(host='0.0.0.0', port=5001, debug=True)
