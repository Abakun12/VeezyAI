import pandas as pd
import logging
from pymongo import MongoClient
import config

logger = logging.getLogger(__name__)

def load_data_from_mongodb():
    """
    Kết nối tới MongoDB và tải dữ liệu từ các collection cần thiết.
    """

    logger.info("Kết nối tới MongoDB...")
    mongo_client = None
    try:
        mongo_client = MongoClient(config.MONGO_CONN_STR, serverSelectionTimeoutMS=5000)
        mongo_client.admin.command('ping')
        db = mongo_client[config.MONGO_DB_NAME]
        logger.info(f"Kết nối tới MongoDB thành công. '{config.MONGO_DB_NAME}'")

        query_filter = {"isActive": True}
        projection = {"eventName": 1, "tags": 1, "categoryId": 1,"isActive": 1}
        events_cursor  = db[config.EVENTS_COLLECTION].find(query_filter, projection)
        events_df = pd.DataFrame(list(events_cursor))
        if not events_df.empty:
            events_df.rename(columns={'_id': 'eventId'}, inplace=True)
            events_df['eventId'] = events_df['eventId'].astype(str)
        logger.info(f"-> Tìm thấy {len(events_df)} document trong collection '{config.EVENTS_COLLECTION}'.")

        users_cursor = db[config.USERS_COLLECTION].find({}, {"accountId": 1, "categories": 1})
        users_df = pd.DataFrame(list(users_cursor))
        if not users_df.empty:
            if 'accountId' in users_df.columns:
                users_df['accountId'] = users_df['accountId'].astype(str)
        logger.info(f"-> Tìm thấy {len(users_df)} document trong collection '{config.USERS_COLLECTION}'.")

        feedback_cursor = db[config.FEEDBACK_COLLECTION].find({}, {"userId": 1, "eventId": 1, "rating": 1})
        feedback_df = pd.DataFrame(list(feedback_cursor))
        if not feedback_df.empty:
            feedback_df.rename(columns={'userId': 'accountId'}, inplace=True)

            feedback_df['accountId'] = feedback_df['AccountId'].astype(str)
            feedback_df['eventId'] = feedback_df['eventId'].astype(str)
        logger.info(f"-> Tìm thấy {len(feedback_df)} document trong collection '{config.FEEDBACK_COLLECTION}'.")

        is_data_missing = events_df.empty or users_df.empty or feedback_df.empty
        if is_data_missing:
            logger.warning("Dữ liệu không có trong collection. Vui lòng thiết lập lại dữ liệu.")
            return load_sample_data()
        logger.info(f"✓ Tải và chuẩn hóa dữ liệu từ MongoDB hoàn tất.")
        return users_df, events_df, feedback_df

    except Exception as e:
        logger.exception(f"LỖI: Không thể tải dữ liệu từ MongoDB: {e}. Sử dụng dữ liệu mẫu thay thế.")
        return load_sample_data()
    finally:
        if mongo_client:
            mongo_client.close()
            logger.info("Đã đóng kết nối MongoDB.")

def load_sample_data():
    """Tạo dữ liệu mẫu thực tế hơn để thử nghiệm."""
    logger.info("Đang tạo dữ liệu mẫu (fallback)...")
    events_data = {
        'EventId': [f'event_{i:02}' for i in range(1, 11)],
        'EventName': [f'Sự kiện Âm nhạc Pop {i}' for i in range(1, 4)] + [f'Hội thảo AI {i}' for i in range(1, 4)] + [f'Triển lãm Nghệ thuật {i}' for i in range(1, 3)] + [f'Đại nhạc hội Rock {i}' for i in range(1, 3)],
        'Tags': [['pop', 'live', 'ho-chi-minh'], ['pop', 'dj'], ['pop', 'acoustic', 'ha-noi'], ['ai', 'cong-nghe', 'khoa-hoc'], ['ai', 'startup'], ['cong-nghe', 'deep-learning', 'ha-noi'], ['nghe-thuat', 'tranh-ve'], ['nghe-thuat', 'hien-dai', 'ho-chi-minh'], ['rock', 'band', 'indie'], ['rock', 'live', 'da-nang']],
        'CategoryIds': [['concert'], ['concert'], ['concert'], ['seminar'], ['seminar'], ['seminar'], ['exhibition'], ['exhibition'], ['concert'], ['concert']]
    }
    events_df = pd.DataFrame(events_data)
    users_data = { 'AccountId': ['user_A', 'user_B', 'user_C', 'user_D_new', 'user_E_new_prefs'], 'Categories': [['concert', 'pop', 'live'], ['seminar', 'ai', 'cong-nghe'], ['rock', 'live', 'nghe-thuat'], [], ['nghe-thuat', 'trien-lam']] }
    users_df = pd.DataFrame(users_data)
    feedback_data = { 'AccountId': ['user_A', 'user_A', 'user_A', 'user_B', 'user_B', 'user_C', 'user_C'], 'EventId': ['event_01', 'event_02', 'event_09', 'event_04', 'event_05', 'event_03', 'event_07'], 'Rating': [5, 4, 5, 5, 4, 5, 3] }
    feedback_df = pd.DataFrame(feedback_data)
    logger.info("✓ Dữ liệu mẫu đã sẵn sàng.")
    return users_df, events_df, feedback_df

def get_reviews_for_event(event_id: str):

    logger.info(f"Đang truy vấn các bình luận cho EventId: {event_id}")
    mongo_client = None
    try:
        mongo_client = MongoClient(config.MONGO_CONN_STR, serverSelectionTimeoutMS=5000)
        db = mongo_client[config.MONGO_DB_NAME]
        comment_collection = db[config.COMMENT_COLLECTION]

        query = {"eventId": event_id, "content": {"$exists": True, "$ne": ""}}
        projection = {"content": 1, "_id": 0}

        reviews_cursor = comment_collection.find(query, projection)
        reviews = [doc.get("content") for doc in reviews_cursor if doc.get("content")]

        logger.info(f"-> Tìm thấy {len(reviews)} bình luận cho EventId: {event_id}")
        return reviews
    except Exception as e:
        logger.exception(f"LỖI: Không thể truy vấn các bình luận cho EventId: {event_id}: {e}")
        return []
    finally:
        if mongo_client:
            mongo_client.close()