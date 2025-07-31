import logging
import threading
import pandas as pd

# Import các thành phần cần thiết từ các module khác
import config
import data_loader
from .content_based import build_user_profile, get_cbf_recommendations, build_event_profiles
from .collaborative import get_cf_recommendations, train_cf_model

logger = logging.getLogger(__name__)

# Các biến toàn cục để giữ mô hình và dữ liệu đã được huấn luyện/tải
EVENTS_DF = None
INTERACTION_DF = None
EVENT_PROFILES = None
CF_MODEL_INFO = None

# Một "khóa" để ngăn chặn nhiều request cùng lúc khởi tạo mô hình
model_lock = threading.Lock()


def load_and_train_models():
    """Tải dữ liệu và huấn luyện các mô hình."""
    logger.info("--- BẮT ĐẦU TẢI VÀ HUẤN LUYỆN MÔ HÌNH CHO WORKER ---")
    try:
        events_df, interaction_df = data_loader.load_data_from_mongodb()
        event_profiles, _ = build_event_profiles(events_df)
        cf_model_info = train_cf_model(interaction_df)
        logger.info("--- KẾT THÚC TẢI VÀ HUẤN LUYỆN MÔ HÌNH ---")
        return events_df, interaction_df, event_profiles, cf_model_info
    except Exception as e:
        logger.exception(f"Lỗi nghiêm trọng khi khởi tạo và huấn luyện mô hình: {e}")
        return None, None, None, None


def ensure_models_are_loaded():
    """Hàm then chốt: Đảm bảo các mô hình đã được tải trong worker process hiện tại."""
    global EVENTS_DF, INTERACTION_DF, EVENT_PROFILES, CF_MODEL_INFO
    if EVENT_PROFILES is None:
        with model_lock:
            if EVENT_PROFILES is None:
                logger.info("Mô hình chưa được tải trong worker này. Bắt đầu quá trình tải...")
                EVENTS_DF, INTERACTION_DF, EVENT_PROFILES, CF_MODEL_INFO = load_and_train_models()
                if EVENT_PROFILES is None:
                    logger.error("!!! Tải mô hình thất bại. Worker sẽ không thể hoạt động đúng. !!!")
                else:
                    logger.info("✓ Mô hình đã được tải thành công cho worker này.")


def get_recommendations_for_user(account_id, top_k=5):
    """Hàm chính để lấy gợi ý cho một người dùng, kết hợp CF và CBF."""
    logger.info(f"Bắt đầu lấy gợi ý cho người dùng: {account_id}")
    if INTERACTION_DF is None or EVENT_PROFILES is None:
        logger.error("Dữ liệu/Mô hình chưa sẵn sàng để tạo gợi ý.")
        return []

    events_user_interacted = INTERACTION_DF[INTERACTION_DF['accountId'] == account_id]['eventId'].unique()

    # Xử lý Cold Start: người dùng mới chưa có tương tác
    if events_user_interacted.size == 0:
        logger.info(f"Người dùng '{account_id}' là người dùng mới (cold start). Fallback: Gợi ý sự kiện phổ biến.")
        if INTERACTION_DF.empty:
            return []  # Không có tương tác nào trong toàn hệ thống
        popularity_scores = INTERACTION_DF['eventId'].value_counts().reset_index()
        popularity_scores.columns = ['eventId', 'attendance_count']
        return popularity_scores.sort_values('attendance_count', ascending=False)['eventId'].head(top_k).tolist()

    final_scores = {}
    alpha = config.HYBRID_ALPHA

    # Lấy gợi ý từ Content-Based
    user_profile_vector = build_user_profile(account_id, INTERACTION_DF, EVENT_PROFILES)
    cbf_recs = get_cbf_recommendations(user_profile_vector, EVENT_PROFILES, events_user_interacted, top_k * 2)
    for event_id, score in cbf_recs.items():
        final_scores[event_id] = final_scores.get(event_id, 0) + alpha * score

    # Lấy gợi ý từ Collaborative Filtering
    cf_recs = get_cf_recommendations(account_id, CF_MODEL_INFO, events_user_interacted, top_k * 2)
    cf_scores = list(cf_recs.values())
    if cf_scores:
        min_score, max_score = min(cf_scores), max(cf_scores)
        if (max_score - min_score) > 0:
            for event_id, score in cf_recs.items():
                norm_score = (score - min_score) / (max_score - min_score)
                final_scores[event_id] = final_scores.get(event_id, 0) + (1 - alpha) * norm_score

    sorted_recs = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
    logger.info(f"✓ Đã tạo xong danh sách gợi ý cho {account_id}.")
    return [event_id for event_id, score in sorted_recs[:top_k]]
