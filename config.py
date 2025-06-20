import os

MONGO_CONN_STR  = os.environ.get("MONGO_CONN_STR", "mongodb+srv://thuanchce170133:ZEQ16jqwjtolxbaV@cluster0.ajszsll.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
MONGO_DB_NAME = os.environ.get("MONGO_DB_NAME", "veezy_db")

ACCOUNTS_COLLECTION = "Account"
USERS_COLLECTION = "User"
EVENTS_COLLECTION = "Event"
TICKET_ORDER_COLLECTION = "TicketOrder"
ATTENDANCE_COLLECTION = "Attendance"
FEEDBACK_COLLECTION = "Feedback"
FOLLOW_COLLECTION = "Follow"


MODEL_NAME = "Facenet"
DETECTOR_NAME = "retinaface"
DISTANCE_METRIC = "cosine"

# --- Collaborative Filtering ---
CF_N_COMPONENTS = 10

# --- Hybrid Model ---
HYBRID_ALPHA = 0.6