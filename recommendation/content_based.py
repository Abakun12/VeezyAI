import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)


def build_event_profiles(events_df):
    """Tạo profile vector cho mỗi sự kiện dựa trên tags và categoryId."""
    if events_df.empty:
        return pd.DataFrame(), None

    logger.info("Đang xây dựng hồ sơ (profile) cho các sự kiện...")
    df = events_df.copy()

    # Xử lý dữ liệu欠損 (missing data) và chuẩn hóa
    df['tags'] = df['tags'].apply(lambda x: x if isinstance(x, list) else [])
    df['categoryId'] = df['categoryId'].apply(lambda x: [x] if isinstance(x, str) else [])

    # Kết hợp tags và categoryId thành một chuỗi văn bản duy nhất
    df['content'] = df.apply(lambda row: ' '.join(row['tags']) + ' ' + ' '.join(row['categoryId']), axis=1)

    # Sử dụng TF-IDF để chuyển đổi văn bản thành vector
    tfidf = TfidfVectorizer(stop_words=None)
    event_profiles_matrix = tfidf.fit_transform(df['content'])

    event_profiles = pd.DataFrame(event_profiles_matrix.toarray(), index=df['eventId'],
                                  columns=tfidf.get_feature_names_out())

    logger.info("✓ Hồ sơ sự kiện đã được tạo.")
    return event_profiles, tfidf


def build_user_profile(account_id, interaction_df, event_profiles):
    """Xây dựng profile vector cho người dùng dựa trên các sự kiện đã tương tác."""
    user_interactions = interaction_df[interaction_df['accountId'] == account_id]
    if user_interactions.empty:
        return None

    interacted_event_ids = user_interactions['eventId'].tolist()
    valid_event_profiles = event_profiles[event_profiles.index.isin(interacted_event_ids)]

    if valid_event_profiles.empty:
        return None

    # Profile của người dùng là vector trung bình của các sự kiện họ đã tương tác
    return valid_event_profiles.mean(axis=0).values


def get_cbf_recommendations(user_profile_vector, event_profiles, events_user_interacted, top_k=10):
    """Lấy gợi ý dựa trên sự tương đồng cosine giữa profile người dùng và profile sự kiện."""
    if user_profile_vector is None:
        return {}

    # Loại bỏ các sự kiện người dùng đã tương tác
    candidate_events = event_profiles.drop(events_user_interacted, errors='ignore')
    if candidate_events.empty:
        return {}

    user_profile_reshaped = user_profile_vector.reshape(1, -1)
    cbf_scores = cosine_similarity(user_profile_reshaped, candidate_events).flatten()

    # Lấy top K sự kiện có điểm cao nhất
    top_indices = cbf_scores.argsort()[::-1][:top_k]
    return {candidate_events.index[i]: cbf_scores[i] for i in top_indices}

