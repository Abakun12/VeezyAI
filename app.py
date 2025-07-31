import os
from flask import Flask, request, jsonify, Response
import logging
import joblib
from flask.cli import load_dotenv
import google.generativeai as genai
import config
import pandas as pd
import data_loader
from bson import ObjectId
from pymongo import MongoClient
from ticket_suggestion_model.prediction_api import get_features_for_suggestion
from face_recognition.face_enroll import enroll_or_update_face, preload_models, buy_ticket_by_face
from face_recognition.face_identification_routes import identify_face_from_image_and_db, identify_face_for_check_in, _initialize_models
from nlp_service.sentiment_analyzer import SentimentAnalyzer
from nlp_service.aspect_analyzer import analyze_aspects
from nlp_service.keyword_extractor import  extract_top_keywords
from recommendation import hybrid

load_dotenv()
app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- CÁC BIẾN TOÀN CỤC ĐỂ LƯU MODEL VÀ DỮ LIỆU ---
USERS_DF = None
EVENTS_DF = None
INTERACTION_DF = None
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

    model = joblib.load('ticket_suggestion_model/saved_models/ticket_suggestion_model.pkl')
    model_columns = joblib.load('ticket_suggestion_model/saved_models/suggestion_model_columns.pkl')

    if not config.MONGO_CONN_STR:
        raise ValueError("Biến môi trường MONGO_CONN_STR chưa được thiết lập.")
    mongo_client = MongoClient(config.MONGO_CONN_STR)
    db = mongo_client[config.MONGO_DB_NAME]
    events_collection = db[config.EVENTS_COLLECTION]
    chunks_collection = db[config.KNOWLEDGE_CHUNKS_COLLECTION]
    logger.info("Đã kết nối Gemini API và MongoDB thành công.")
except Exception as e:
    logger.error(f"Lỗi khi thiết lập cấu hình Gemini API: {e}")

@app.route('/')
def index():
    return "Recommendation Service for Veezy AI is running."

@app.route('/ai/recommend', methods=['POST'])
def recommend_for_user_api():
    """API endpoint chính để nhận yêu cầu và trả về gợi ý."""

    # BƯỚC 1: Đảm bảo mô hình đã được tải trong worker này.
    hybrid.ensure_models_are_loaded()

    # BƯỚC 2: Kiểm tra lại xem quá trình tải có thành công không.
    # SỬA LỖI: Truy cập các biến thông qua module 'hybrid'
    if hybrid.EVENTS_DF is None or hybrid.INTERACTION_DF is None:
        logger.error("Tải mô hình/dữ liệu thất bại. Dịch vụ chưa sẵn sàng.")
        return jsonify({"error": "Service is not ready, please try again later."}), 503

    # BƯỚC 3: Xử lý request JSON từ client
    try:
        request_data = request.get_json()
        if not request_data or 'account_id' not in request_data:
            return jsonify({"error": "Missing 'account_id' in JSON body."}), 400
        account_id = request_data.get('account_id')
        top_k = 5
    except Exception:
            return jsonify({"error": "Invalid JSON request."}), 400

    logger.info(f"Nhận được yêu cầu gợi ý cho người dùng: {account_id} với top_k={top_k}")

    # SỬA LỖI: Xóa bỏ đoạn kiểm tra không cần thiết gây ra KeyError.
    # Logic xử lý người dùng mới (cold start) đã có trong hàm get_recommendations_for_user.

    # BƯỚC 4: Gọi hàm logic để lấy gợi ý và trả về kết quả
    try:
        recommendations_ids = hybrid.get_recommendations_for_user(account_id, top_k=top_k)
        if not recommendations_ids:
            return jsonify({"account_id": account_id, "recommendations": []})

        # Lấy chi tiết sự kiện từ EVENTS_DF toàn cục
        # SỬA LỖI: Truy cập biến thông qua module 'hybrid'
        recommended_events_details = hybrid.EVENTS_DF[hybrid.EVENTS_DF['eventId'].isin(recommendations_ids)].to_dict(
            orient='records')

        # Sắp xếp kết quả theo đúng thứ tự đã gợi ý
        ordered_results = sorted(recommended_events_details, key=lambda x: recommendations_ids.index(x['eventId']))

        return jsonify({"account_id": account_id, "recommendations": ordered_results})
    except Exception as e:
        logger.exception(f"Lỗi khi tạo gợi ý cho người dùng {account_id}: {e}")
        return jsonify({"error": "An internal error occurred while generating the suggestion."}), 500


@app.route('/ai/enroll', methods=['POST'])
def enroll_face_endpoint():
    return enroll_or_update_face()

@app.route('/ai/buy_ticket_enroll_face', methods=['POST'])
def buy_ticket_by_face_endpoint():
    return buy_ticket_by_face()

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
            return jsonify({"error": "Missing 'event_id' in JSON body."}), 400
        event_id = request_data['event_id']
    except Exception:
        return jsonify({"error": "Invalid JSON request."}), 400

    logger.info(f"Nhận được yêu cầu phân tích cảm xúc cho EventId: {event_id}")

    reviews = data_loader.get_reviews_for_event(event_id)
    if not reviews:
        return jsonify({"error": f"No comments found for EventId: {event_id}."}), 404

    sentiment_results = sentiment_analyzer.analyze(reviews)
    if not sentiment_results:
            return jsonify({"error": "Error in sentiment analysis."}), 500

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

@app.route('/ai/ingest-knowledge', methods=['POST'])
def ingest_knowledge():
    """
    Endpoint này được Backend gọi khi một sự kiện được tạo/cập nhật.
    AI Service sẽ tự truy vấn DB để xây dựng cơ sở tri thức.
    """
    if not mongo_client:
        return Response("Service error: Unable to connect to DB.", status=503)

    data = request.get_json()
    if not data or 'eventId' not in data:
        return jsonify({"error": "Missing eventId"}), 400

    event_id_str = data['eventId']
    logger.info(f"Starting knowledge ingestion for eventId: {event_id_str}")

    try:
        event_doc = events_collection.find_one({"_id": event_id_str})
        if not event_doc:
            return jsonify({"error": f"Event not found for eventId: {event_id_str}"}), 404

        knowledge_text = ""
        knowledge_text += f"Event Name: {event_doc.get('eventName', 'Not available')}. "
        knowledge_text += f"Description: {event_doc.get('eventDescription', 'Not available')}. "
        knowledge_text += f"Location: {event_doc.get('eventLocation', 'Not available')}. "
        knowledge_text += f"Start Time: {event_doc.get('startAt')}. "
        knowledge_text += f"End Time: {event_doc.get('endAt')}. "

        if event_doc.get('bankName'):
            knowledge_text += f"Payment Information: Bank {event_doc.get('bankName')}, "
            knowledge_text += f"account number {event_doc.get('bankAccount')}, "
            knowledge_text += f"account name {event_doc.get('bankAccountName')}. "

        tags = event_doc.get('tags', [])
        if tags:
            knowledge_text += f"Related topics: {', '.join(tags)}. "

        text_chunks = chunk_text(knowledge_text)
        if not text_chunks:
            return jsonify({"message": "No content to process."}), 200

        embedding_result = genai.embed_content(model="models/text-embedding-004", content=text_chunks,
                                               task_type="RETRIEVAL_DOCUMENT")
        embeddings = embedding_result['embedding']

        chunks_collection.delete_many({"eventId": event_id_str})

        chunks_to_insert = [{
            "eventId": event_id_str,
            "content": chunk_content,
            "embedding": embeddings[i]
        } for i, chunk_content in enumerate(text_chunks)]

        if chunks_to_insert:
            chunks_collection.insert_many(chunks_to_insert)

        return jsonify({"status": "success", "chunks_created": len(chunks_to_insert)}), 200

    except Exception as e:
        logger.exception(f"Error during knowledge ingestion: {e}")
        return jsonify({"error": "Error on AI service while processing document."}), 500

@app.route('/ai/process-chat-request-stream', methods=['POST'])
def process_chat_request_stream():
    """
    Endpoint chính để xử lý câu hỏi của người dùng.
    """
    if not mongo_client:
        return Response("Service error: Unable to connect to DB.", status=503)

    data = request.get_json()
    if not data or 'user_question' not in data:
        return jsonify({"error": "Missing user_question"}), 400

    user_question = data['user_question']
    event_id = data.get('eventId')

    try:
        if not event_id:
            all_events = list(events_collection.find({}, {"eventName": 1}))
            event_names = [event.get("eventName") for event in all_events if event.get("eventName")]

            ner_model = genai.GenerativeModel('gemini-1.5-flash')
            ner_prompt = f"From the following list of event names: {event_names}. Which event name is mentioned in the question: \"{user_question}\"? Return the exact name, or 'None' if not found."
            ner_response = ner_model.generate_content(ner_prompt)
            found_event_name = ner_response.text.strip()

            if found_event_name != 'None':
                found_event = events_collection.find_one({"eventName": found_event_name})
                if found_event:
                    event_id = str(found_event.get("_id"))
            else:
                return Response("To better assist you, please specify which event you are asking about.",
                                mimetype='text/plain; charset=utf-8')
        logger.info(f"Using eventId for vector search filter: {event_id}")
        question_embedding = genai.embed_content(
            model="models/text-embedding-004",
            content=[user_question],
            task_type="RETRIEVAL_QUERY"
        )['embedding'][0]

        pipeline = [
            {
                "$search": {
                    "index": "default",  # Tên index bạn vừa tạo ở trên
                    "text": {
                        "query": user_question,  # Dùng câu hỏi của người dùng để tìm kiếm
                        "path": "content"  # Tìm trong trường "content" của collection chunks
                    }
                }
            },
            {
                "$limit": 3  # Lấy 3 kết quả phù hợp nhất
            },
            {
                "$project": {
                    "content": 1,
                    "score": {"$meta": "searchScore"}
                }
            }
        ]
        relevant_chunks = list(chunks_collection.aggregate(pipeline))
        context = "\n\n".join(
            [chunk.get("content", "") for chunk in relevant_chunks]) or "No detailed information found."

        generation_model = genai.GenerativeModel('gemini-1.5-flash')
        final_prompt = f"Based on the following information: \"{context}\". Answer the question: \"{user_question}\""

        response_stream = generation_model.generate_content(final_prompt, stream=True)

        def generate():
            for chunk in response_stream:
                if chunk.text:
                    yield chunk.text

        logger.info(f"Streamed response generated for question: '{user_question}'")
        return Response(generate(), mimetype='text/plain; charset=utf-8')

    except Exception as e:
        logger.exception(f"Error during chat processing: {e}")
        return Response("An error occurred during processing.", status=500)

@app.route('/ai/suggest-quantity', methods=['POST'])
def suggest_quantity():
    if model is None:
        return jsonify({"error": "Model is not loaded"}), 500

    json_data = request.get_json()
    if not json_data or 'eventId' not in json_data:
        return jsonify({"error": "Request body must contain 'eventId'"}), 400

    event_id = json_data['eventId']

    try:
        # Lấy toàn bộ features từ DB
        features = get_features_for_suggestion(event_id)
        if features is None:
            return jsonify({"error": f"Could not retrieve data for event_id {event_id}"}), 404

        # Chuẩn bị dữ liệu và dự đoán
        new_event_df = pd.DataFrame([features])
        new_event_encoded = pd.get_dummies(new_event_df, drop_first=True)
        final_new_event = new_event_encoded.reindex(columns=model_columns, fill_value=0)
        prediction_quantity = model.predict(final_new_event)

        # Trả về kết quả
        return jsonify({"suggested_quantity": int(prediction_quantity[0])})

    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        return jsonify({"error": "An internal error occurred."}), 500

preload_models()
_initialize_models()
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=False)
