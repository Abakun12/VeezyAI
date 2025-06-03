import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def get_similarity_matrix(interactions_df):
    """
    Tính ma trận tương đồng giữa các người dùng dựa trên lịch sử tương tác.

    Args:
        interactions_df (pd.DataFrame): DataFrame chứa dữ liệu tương tác (userId, eventId).

    Returns:
        pd.DataFrame: Ma trận tương đồng giữa các người dùng, hoặc DataFrame rỗng nếu không có dữ liệu.
    """
    if interactions_df.empty:
        return pd.DataFrame()

    user_event_matrix = pd.crosstab(interactions_df['userId'], interactions_df['eventId'])
    similarities = cosine_similarity(user_event_matrix.fillna(0))
    user_ids = user_event_matrix.index.tolist()
    return pd.DataFrame(similarities, index=user_ids, columns=user_ids)

def get_cf_recommendations(user_id, sim_df, interactions_df, top_k):
    """
    Lấy gợi ý từ Collaborative Filtering dựa trên người dùng tương đồng.

    Args:
        user_id (str): ID của người dùng cần gợi ý.
        sim_df (pd.DataFrame): Ma trận tương đồng giữa các người dùng.
        interactions_df (pd.DataFrame): DataFrame chứa dữ liệu tương tác.
        top_k (int): Số lượng gợi ý tối đa.

    Returns:
        list: Danh sách eventId được gợi ý từ CF.
    """
    if sim_df.empty or user_id not in sim_df.columns:
        return []

    top_sim_users = sim_df[user_id].sort_values(ascending=False)[1:top_k+1].index.tolist()
    events_user = set(interactions_df[interactions_df["userId"] == user_id]["eventId"])
    candidate_events = interactions_df[interactions_df["userId"].isin(top_sim_users)]["eventId"].unique()
    cf_suggestions = [e for e in candidate_events if e not in events_user]
    return cf_suggestions
