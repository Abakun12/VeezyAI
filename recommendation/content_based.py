import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)

def build_event_profiles(events_df):
    """
    Tạo profile của các event từ dữ liệu.
    """
    if events_df.empty or 'tags' not in events_df.columns or 'categoryId' not in events_df.columns:
        logger.warning("DataFrame sự kiện không hợp lệ để xây dựng hồ sơ.")
        return pd.DataFrame(), None
    logger.info("Xây dựng hồ sơ của các event...")
    df = events_df.copy()
    df['content'] = df['tags'].apply(lambda row: ' '.join(row.get('tags', [])) + ' ' + ' '.join(row.get('categoryId', [])))
    tfidf = TfidfVectorizer(stop_words='None')
    event_profiles_matrix = tfidf.fit_transform(df['content'])
    event_profiles = pd.DataFrame(event_profiles_matrix.toarray(), index=df['evenId'], columns=tfidf.get_feature_names_out())
    logger.info("✓ Hồ sơ sự kiện đã được tạo.")
    return event_profiles, tfidf

def build_user_profiles(account_id, feedback_df, event_profiles):
    """
    Tạo profile của các user từ dữ liệu.
    """
    user_interactions = feedback_df[feedback_df['accountId'] == account_id]
    if user_interactions.empty: return None
    high_rating_interactions = user_interactions[user_interactions['rating'] >= 4]
    if high_rating_interactions.empty: return None
    interacted_event_profiles = event_profiles.loc[high_rating_interactions['eventId']]
    user_profile_vector = interacted_event_profiles.mean(axis=0).values
    return user_profile_vector

def get_cbf_recommendations(user_profile_vector,event_profiles, events_user_interacted, top_k=10):
    if user_profile_vector is None: return {}
    candidate_envets = event_profiles.drop(events_user_interacted, errors='ignore')
    if candidate_envets.empty: return {}
    user_profile_reshaped = user_profile_vector.reshape(1, -1)
    cbf_scores = cosine_similarity(user_profile_reshaped, candidate_envets).flatten()
    top_indices = cbf_scores.argsort()[::-1][:top_k]
    return { candidate_envets.index[i]: cbf_scores[i] for i in top_indices }
