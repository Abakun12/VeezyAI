import os
from flask import Flask, request, jsonify, Response
import logging
from flask.cli import load_dotenv
import google.generativeai as genai
import config
import data_loader
from bson import ObjectId
from pymongo import MongoClient
from recommendation.hybrid import get_recommendations_for_user, initialize_and_train
from face_recognition.face_enroll import enroll_face, preload_models
from face_recognition.verify_face import verify_face
from face_recognition.face_identification_routes import identify_face_from_image_and_db, identify_face_for_check_in
from nlp_service.sentiment_analyzer import SentimentAnalyzer
from nlp_service.aspect_analyzer import analyze_aspects
from nlp_service.keyword_extractor import  extract_top_keywords

load_dotenv()
app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- CÁC BIẾN TOÀN CỤC ĐỂ LƯU MODEL VÀ DỮ LIỆU ---
USERS_DF = None
EVENTS_DF = None
FEEDBACK_DF = None
EVENT_PROFILES = None
CF_MODEL_INFO = None

logger.info("--- BẮT ĐẦU PHA OFFLINE (KHỞI ĐỘNG SERVER) ---")
sentiment_analyzer = SentimentAnalyzer()
logger.info("--- KẾT THÚC PHA OFFLINE. SERVER SẴN SÀNG NHẬN REQUEST. ---")

# --- Cấu hình Gemini API ---
try:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Cấu hình Gemini API không hợp lệ.")
    genai.configure(api_key=api_key)
    logger.info("Cấu hình Gemini API đã được thiết lập.")
    if not config.MONGO_CONN_STR:
        raise ValueError("Biến môi trường MONGO_CONN_STR chưa được thiết lập.")
    mongo_client = MongoClient(config.MONGO_CONN_STR)
    db = mongo_client[config.MONGO_DB_NAME]
    events_collection = db[config.EVENTS_COLLECTION]
    chunks_collection = db[config.KNOWLEDGE_CHUNKS_COLLECTION]
    logger.info("Đã kết nối Gemini API và MongoDB thành công.")
except Exception as e:
    logger.error(f"Lỗi khi thiết lập cấu hình Gemini API: {e}")

@app.before_request
def setup_models():
    initialize_and_train()

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

# @app.route('/ai/verify', methods=['POST'])
# def verify_face_endpoint():
#     return verify_face()

@app.route('/ai/identify', methods=['POST'])
def identify_face_endpoint():
    return identify_face_from_image_and_db()

@app.route("/ai/check_in_face", methods=['POST'])
def handle_check_in_face():
    return identify_face_for_check_in()

@app.route('/ai/analyze_sentiment', methods=['POST'])
def analyze_sentiment_endpoint():
    if not sentiment_analyzer or not sentiment_analyzer.sentiment_pipeline:
        return jsonify({"error": "Service is not ready, please try again later."}), 503

    try:
        request_data = request.get_json()
        if not request_data or 'event_id' not in request_data:
            return jsonify({"error": "Thiếu 'event_id' trong JSON body."}), 400
        event_id = request_data['event_id']
    except Exception:
        return jsonify({"error": "Request JSON không hợp lệ."}), 400

    logger.info(f"Nhận được yêu cầu phân tích cảm xúc cho EventId: {event_id}")

    reviews = data_loader.get_reviews_for_event(event_id)
    if not reviews:
        return jsonify({"message": f"Không tìm thấy bình luận cho EventId: {event_id}."}), 404

    sentiment_results = sentiment_analyzer.analyze(reviews)
    if not sentiment_results:
        return jsonify({"error": "Lỗi trong quá trình phân tích cảm xúc."}), 500

    classified_reviews = []
    positive_count, negative_count, neutral_count = 0, 0, 0
    for review, result in zip(reviews, sentiment_results):
        label = result['label']

        if label == 'POSITIVE': positive_count += 1
        elif label == 'NEGATIVE': negative_count += 1
        else: neutral_count += 1

        classified_reviews.append({
            "text": review,
            "sentiment": label,
            "score": result['score']
        })
    aspect_sentiments = analyze_aspects(reviews, sentiment_analyzer)

    top_keywords = extract_top_keywords(classified_reviews)

    negative_review_sorted = sorted([r for r in classified_reviews if r['sentiment'] == 'NEGATIVE'], key=lambda x: x['score'])
    top_negative_reviews = [
        {"text": r['text'],
         "socre": round(r['score'], 2)} for r in negative_review_sorted[:5]
    ]

    total_reviews = len(reviews)

    final_output = {
        "overall_sentiment": {
            "positive_percentage": round((positive_count / total_reviews) * 100 , 1),
            "negative_percentage": round((negative_count / total_reviews) * 100 , 1),
            "neutral_percentage": round((neutral_count / total_reviews) * 100 , 1)
        },
        "aspect_sentiments": aspect_sentiments,
        "top_keywords": top_keywords,
        "negative_reviews": top_negative_reviews
    }

    return jsonify(final_output)

def chunk_text(text, max_words=200):
    """Hàm trợ giúp để chia nhỏ văn bản."""
    words = text.split()
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]

@app.route('/ingest-knowledge', methods=['POST'])
def ingest_knowledge():
    """
    Endpoint này được Backend gọi khi một sự kiện được tạo/cập nhật.
    AI Service sẽ tự truy vấn DB để xây dựng cơ sở tri thức.
    """
    if not mongo_client: return Response("Lỗi dịch vụ: Không thể kết nối DB.", status=503)

    data = request.get_json()
    if not data or 'eventId' not in data:
        return jsonify({"error": "Thiếu eventId"}), 400

    event_id_str = data['eventId']
    logger.info(f"Bắt đầu xử lý kiến thức cho eventId: {event_id_str}")

    try:
        # 1. AI Service TỰ TRUY VẤN DB
        event_doc = events_collection.find_one({"_id": ObjectId(event_id_str)})
        if not event_doc or "eventDescription" not in event_doc:
            return jsonify({"error": f"Không tìm thấy sự kiện hoặc mô tả cho eventId: {event_id_str}"}), 404

        # 2. Chia nhỏ và Vector hóa
        text_chunks = chunk_text(event_doc["eventDescription"])
        if not text_chunks:
            return jsonify({"message": "Không có nội dung để xử lý."}), 200

        embedding_result = genai.embed_content(model="models/text-embedding-004", content=text_chunks,
                                               task_type="RETRIEVAL_DOCUMENT")
        embeddings = embedding_result['embedding']

        # 3. Lưu vào DB
        chunks_collection.delete_many({"EventId": ObjectId(event_id_str)})
        chunks_to_insert = [{
            "EventId": ObjectId(event_id_str),
            "Content": chunk_text,
            "Embedding": embeddings[i]
        } for i, chunk_text in enumerate(text_chunks)]

        if chunks_to_insert:
            chunks_collection.insert_many(chunks_to_insert)

        return jsonify({"status": "success", "chunks_created": len(chunks_to_insert)}), 200

    except Exception as e:
        logger.exception(f"Lỗi khi xử lý kiến thức: {e}")
        return jsonify({"error": "Lỗi phía AI service khi xử lý tài liệu."}), 500

@app.route('/process-chat-request-stream', methods=['POST'])
def process_chat_request_stream():
    """
    Endpoint chính để xử lý câu hỏi của người dùng.
    """
    if not mongo_client: return Response("Lỗi dịch vụ: Không thể kết nối DB.", status=503)

    data = request.get_json()
    if not data or 'user_question' not in data:
        return jsonify({"error": "Thiếu user_question"}), 400

    user_question = data['user_question']
    event_id = data.get('eventId')

    try:
        # Bước 1: Xác định EventID nếu chưa có
        if not event_id:
            all_events = list(events_collection.find({}, {"EventName": 1}))
            event_names = [event.get("EventName") for event in all_events if event.get("EventName")]

            ner_model = genai.GenerativeModel('gemini-1.5-flash')
            ner_prompt = f"Từ danh sách tên sự kiện sau: {event_names}. Tên sự kiện nào được nhắc đến trong câu hỏi: \"{user_question}\"? Trả về chính xác tên đó, hoặc 'None' nếu không có."
            ner_response = ner_model.generate_content(ner_prompt)
            found_event_name = ner_response.text.strip()

            if found_event_name != 'None':
                found_event = events_collection.find_one({"EventName": found_event_name})
                if found_event: event_id = str(found_event.get("_id"))
            else:
                return Response("Để hỗ trợ tốt hơn, bạn vui lòng cho biết bạn đang hỏi về sự kiện nào ạ?",
                                mimetype='text/plain; charset=utf-8')

        # Bước 2: Truy xuất thông tin (Retrieval)
        question_embedding = \
        genai.embed_content(model="models/text-embedding-004", content=[user_question], task_type="RETRIEVAL_QUERY")[
            'embedding'][0]

        pipeline = [
            {"$vectorSearch": {"index": "knowledge_vector_index", "path": "Embedding",
                               "queryVector": question_embedding, "numCandidates": 100, "limit": 3}},
            {"$match": {"EventId": ObjectId(event_id)}},
            {"$project": {"Content": 1}}
        ]
        relevant_chunks = list(chunks_collection.aggregate(pipeline))
        context = "\n\n".join(
            [chunk.get("Content", "") for chunk in relevant_chunks]) or "Không tìm thấy thông tin chi tiết."

        # Bước 3: Sinh câu trả lời (Generation)
        generation_model = genai.GenerativeModel('gemini-1.5-flash')
        final_prompt = f"Dựa vào thông tin sau: \"{context}\". Hãy trả lời câu hỏi: \"{user_question}\""

        response_stream = generation_model.generate_content(final_prompt, stream=True)

        def generate():
            for chunk in response_stream:
                if chunk.text: yield chunk.text

        return Response(generate(), mimetype='text/plain; charset=utf-8')

    except Exception as e:
        logger.exception(f"Lỗi trong quá trình xử lý chat: {e}")
        return Response("Đã xảy ra lỗi trong quá trình xử lý.", status=500)


if __name__ == '__main__':
    initialize_and_train()
    preload_models()
    app.run(host='0.0.0.0', port=5001, debug=False)
