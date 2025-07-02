import pandas as pd
import numpy as np
import logging
import config
from .content_based import build_user_profiles, get_cbf_recommendations, build_event_profiles
from .collaborative import get_cf_recommendations_sklearn, train_cf_model_sklearn
import data_loader
logger = logging.getLogger(__name__)


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

def get_recommendations_for_user(account_id, users_df, events_df, feedback_df, event_profiles, cf_model_info, top_k=5):
    utility_matrix = cf_model_info.get('utility_matrix')
    events_user_interacted = feedback_df[feedback_df['accountId'] == account_id]['eventId'].unique()

    if utility_matrix is None or account_id not in utility_matrix.index:
        logger.info(f"Người dùng '{account_id}' là người dùng mới (cold start).")
        logger.info("Fallback: Gợi ý các sự kiện phổ biến nhất.")
        popularity_scores = feedback_df.groupby('eventId')['rating'].agg(['mean', 'count']).reset_index()
        popularity_scores['popularity_score'] = popularity_scores['mean'] * np.log1p(popularity_scores['count'])
        popular_events = popularity_scores.sort_values('popularity_score', ascending=False)
        top_popular = popular_events[~popular_events['eventId'].isin(events_user_interacted)]
        return top_popular['eventId'].head(top_k).to_list()

    # --- NẾU KHÔNG PHẢI COLD START, THỰC HIỆN HYBRID ---
    final_scores = {}
    alpha = config.HYBRID_ALPHA

    user_profile_vector = build_user_profiles(account_id, feedback_df, event_profiles)
    cbf_recs = get_cbf_recommendations(user_profile_vector, event_profiles, events_user_interacted, top_k * 2)
    for event_id, score in cbf_recs.items():
        final_scores[event_id] = final_scores.get(event_id, 0) + alpha * score

    cf_recs = get_cf_recommendations_sklearn(account_id, cf_model_info, events_user_interacted, top_k * 2)
    cf_scores = list(cf_recs.values())
    if cf_scores:
        min_score, max_score = min(cf_scores), max(cf_scores)
        for event_id, score in cf_recs.items():
            norm_score = (score - min_score) / (max_score - min_score) if (max_score - min_score) > 0 else 0
            final_scores[event_id] = final_scores.get(event_id, 0) + (1 - alpha) * norm_score
    sorted_recs = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
    return [event_id for event_id, score in sorted_recs[:top_k]]