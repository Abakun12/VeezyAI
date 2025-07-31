import joblib
import config
from pymongo import MongoClient

try:
    client = MongoClient(config.MONGO_CONN_STR)
    db = client[config.MONGO_DB_NAME]
    print("Đã kết nối tới MongoDB!")
except Exception as e:
    print(f"Lỗi kết nối MongoDB: {e}")
    db = None

category_avg_map = joblib.load('ticket_suggestion_model/saved_models/category_avg_map.pkl')
city_avg_map = joblib.load('ticket_suggestion_model/saved_models/city_avg_map.pkl')
# Giá trị trung bình chung (dùng cho trường hợp mới)
overall_mean_quantity = category_avg_map.mean()

def _parse_location(location_string):
    """Hàm phụ để tách City và Country từ một chuỗi địa điểm."""
    parts = str(location_string).split(',')
    if len(parts) > 1: return parts[0].strip(), parts[-1].strip()
    return location_string, "Unknown"


def get_features_for_suggestion(event_id):
    """
    Hàm này nhận eventId, truy vấn DB và tạo ra bộ đặc trưng hoàn chỉnh.
    """
    if not db: raise ConnectionError("Không thể kết nối tới database.")
    try:
        pipeline = [
            {"$match": {"_id": event_id}},
            {"$lookup": {"from": config.NEWS_COLLECTION, "localField": "_id", "foreignField": "eventId", "as": "event_news"}},
            {"$lookup": {"from": config.FOLLOWS_COLLECTION, "localField": "createdBy", "foreignField": "managerId",
                         "as": "event_followers"}},
            {"$project": {
                "_id": 0, "eventLocation": 1, "startAt": 1, "endAt": 1, "categoryId": 1,
                "NewsCount": {"$size": "$event_news"},
                "FollowerCount": {"$size": "$event_followers"}
            }}
        ]

        data = list(db[config.EVENTS_COLLECTION].aggregate(pipeline))[0]

        category_name = "Unknown"
        if data.get("categoryId"):
            category_data = db[config.CATEGORIES_COLLECTION].find_one({"_id": data["categoryId"][0]})
            if category_data: category_name = category_data.get("categoryName", "Unknown")

        features = {}
        start_at, end_at = data['startAt'], data['endAt']

        features['Duration_Hours'] = round((end_at - start_at).total_seconds() / 3600, 2)
        features['DayOfWeek'] = "Weekend" if start_at.weekday() >= 5 else "Weekday"
        features['TimeOfDay'] = "Evening" if start_at.hour >= 18 else "Afternoon" if start_at.hour >= 12 else "Morning"

        city, country = _parse_location(data.get("eventLocation"))
        features['City'] = city
        features['Country'] = country
        features['Category'] = category_name

        features['EventManagerFollowers'] = data.get("FollowerCount", 0)
        features['NewsCount'] = data.get("NewsCount", 0)

        features['Category_Avg_Quantity'] = category_avg_map.get(category_name, overall_mean_quantity)
        features['City_Avg_Quantity'] = city_avg_map.get(city, overall_mean_quantity)

        return features

    except Exception as e:
        print(f"Lỗi khi lấy features từ DB cho event {event_id}: {e}")
        return None
