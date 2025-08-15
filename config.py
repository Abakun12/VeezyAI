import os

MONGO_CONN_STR  = os.environ.get("MONGO_CONN_STR", os.getenv("MONGO_CONN_API"))
MONGO_DB_NAME = os.environ.get("MONGO_DB_NAME", "veezy")

ACCOUNTS_COLLECTION = "Accounts"
USERS_COLLECTION = "Users"
EVENTS_COLLECTION = "Events"
TICKET_ORDER_COLLECTION = "TicketOrders"
ATTENDANCE_COLLECTION = "Attendances"
FEEDBACK_COLLECTION = "Feedbacks"
FOLLOW_COLLECTION = "Follows"
COMMENT_COLLECTION = "Comments"
KNOWLEDGE_CHUNKS_COLLECTION = "KnowledgeChunks"
FACE_LOGS_COLLECTION = "FaceLogs"
NEWS_COLLECTION = "News"
FOLLOWS_COLLECTION = "Follows"
CATEGORIES_COLLECTION = "Categories"


MODEL_DIR = "saved_models"
MODEL_FILE = "ticket_suggestion_model.pkl"
COLUMNS_FILE = "suggestion_model_columns.pkl"
CATEGORY_MAP_FILE = "category_avg_map.pkl"
CITY_MAP_FILE = "city_avg_map.pkl"

# --- Cấu hình Model recognition face ---
MODEL_NAME = "Facenet"
DETECTOR_NAME = "retinaface"
DISTANCE_METRIC = "cosine"

THRESHOLD = 0.40

# --- Collaborative Filtering ---
CF_N_COMPONENTS = 10

# --- Hybrid Model ---
HYBRID_ALPHA = 0.6

# --- Cấu hình NLP ---
# Model từ Hugging Face được fine-tune cho phân tích cảm xúc
SENTIMENT_MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

ASPECT_KEYWORDS = {
    # Khía cạnh về Âm thanh
    "Sound": ["sound", "audio", "music", "loud", "bass", "clear", "song", "volume", "acoustics", "microphone", "mic",
              "speakers", "muffled", "echo", "crisp"],

    # Khía cạnh về Công tác Tổ chức
    "Organization": ["check-in", "queue", "staff", "support", "security", "organized", "messy", "schedule", "entry",
                     "exit", "flow", "smooth", "chaotic", "management", "volunteers", "friendly", "helpful", "rude",
                     "unhelpful"],

    # Khía cạnh về Địa điểm
    "Venue": ["venue", "place", "seating", "stage", "space", "parking", "clean", "location", "atmosphere", "restrooms",
              "bathrooms", "toilets", "facilities", "signs", "directions", "layout", "temperature", "air conditioning",
              "ac", "comfortable", "uncomfortable"],

    # Khía cạnh về Giá cả
    "Price": ["price", "ticket", "cost", "expensive", "cheap", "value", "fee", "overpriced", "affordable", "worth it",
              "rip-off", "deal", "booking", "purchase"],

    # Khía cạnh về Màn trình diễn / Nội dung chính
    "Performance/Content": ["performance", "artist", "band", "singer", "speaker", "actor", "act", "show", "setlist",
                            "content", "presentation", "engaging", "boring"],

    # Khía cạnh về Đồ ăn và Thức uống
    "Food & Beverage": ["food", "drink", "beverage", "bar", "options", "selection", "taste", "delicious", "quality",
                        "menu"],

    # Khía cạnh về Hình ảnh và Sản xuất
    "Visuals/Production": ["lights", "lighting", "visuals", "screen", "video", "effects", "production"]
}

# File: config.py

# Cấu hình cho model tùy chỉnh
CUSTOM_MODEL_PATH = "face_recognition/arcface_backbone.pth"
# Kích thước embedding của InceptionResnetV1 là 512
EMBEDDING_DIMENSION = 512

# --- THÔNG SỐ QUAN TRỌNG NHẤT TỪ FILE HUẤN LUYỆN ---
# 1. ĐO LƯỜNG KHOẢNG CÁCH
DISTANCE_METRIC_1 = 'euclidean' # Phải khớp với `euclidean` bạn dùng

# 2. NGƯỠNG ĐĂNG NHẬP
# Lấy giá trị `best_threshold` bạn tìm được. Ví dụ: 0.9543
CUSTOM_LOGIN_THRESHOLD = 0.3

# 3. NGƯỠNG ĐĂNG KÝ
# Nên nghiêm ngặt hơn (nhỏ hơn) ngưỡng đăng nhập một chút để đảm bảo tính duy nhất.
# Bạn có thể điều chỉnh giá trị này.
CUSTOM_ENROLL_THRESHOLD = 0.23

# Cấu hình khác
DETECTOR_NAME_1 = "mtcnn"

MIN_BRIGHTNESS_THRESHOLD = 70  # Yêu cầu độ tự tin phát hiện mặt phải trên 90%
MIN_BLUR_THRESHOLD = 14