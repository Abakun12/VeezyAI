import pandas as pd
import logging
from pymongo import MongoClient
import config

logger = logging.getLogger(__name__)
def load_sample_data():
    """Tải dữ liệu mẫu trong trường hợp không kết nối được MongoDB."""
    logger.info("Đang tải dữ liệu mẫu (fallback)...")
    # Trả về các DataFrame rỗng để các hàm xử lý sau đó không bị lỗi
    return pd.DataFrame(), pd.DataFrame()

def load_data_from_mongodb():
    """
    Kết nối tới MongoDB và tải dữ liệu từ các collection cần thiết.
    """
    logger.info("Đang kết nối tới MongoDB...")
    mongo_client = None
    try:
        mongo_client = MongoClient(config.MONGO_CONN_STR, serverSelectionTimeoutMS=5000)
        mongo_client.admin.command('ping')
        db = mongo_client[config.MONGO_DB_NAME]
        logger.info(f"Kết nối tới MongoDB thành công. DB: '{config.MONGO_DB_NAME}'")

        # Tải dữ liệu Events
        events_cursor = db[config.EVENTS_COLLECTION].find(
            {"isActive": True},
            {"_id": 1, "eventName": 1, "tags": 1, "categoryId": 1 , "eventCoverImageUrl": 1}
        )
        events_df = pd.DataFrame(list(events_cursor))
        if not events_df.empty:
            events_df.rename(columns={'_id': 'eventId'}, inplace=True)
            events_df['eventId'] = events_df['eventId'].astype(str)
        logger.info(f"-> Tìm thấy {len(events_df)} document trong collection '{config.EVENTS_COLLECTION}'.")

        # Tải dữ liệu tương tác từ Attendance
        interaction_cursor = db[config.ATTENDANCE_COLLECTION].find(
            {"joinedAt": {"$ne": None}},
            {"accountId": 1, "eventId": 1}
        )
        interaction_df = pd.DataFrame(list(interaction_cursor))
        if not interaction_df.empty:
            interaction_df['accountId'] = interaction_df['accountId'].astype(str)
            interaction_df['eventId'] = interaction_df['eventId'].astype(str)
        logger.info(f"-> Tìm thấy {len(interaction_df)} tương tác trong collection '{config.ATTENDANCE_COLLECTION}'.")

        if events_df.empty or interaction_df.empty:
            logger.warning("Một trong các collection cần thiết không có dữ liệu. Sử dụng dữ liệu mẫu.")
            return load_sample_data()

        logger.info("✓ Tải và chuẩn hóa dữ liệu từ MongoDB hoàn tất.")
        return events_df, interaction_df

    except Exception as e:
        logger.exception(f"LỖI: Không thể tải dữ liệu từ MongoDB: {e}. Sử dụng dữ liệu mẫu thay thế.")
        return load_sample_data()
    finally:
        if mongo_client:
            mongo_client.close()
            logger.info("Đã đóng kết nối MongoDB.")

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