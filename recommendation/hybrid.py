from .content_based import get_cbf_scores
from .collaborative import get_similarity_matrix, get_cf_recommendations

def get_hybrid_recommendations(user_id, active_events, interactions_df, user_preferences, top_k, alpha=0.6):
    """
    Tạo gợi ý sự kiện bằng mô hình Hybrid (Content-Based + Collaborative Filtering).

    Args:
        user_id (str): ID của người dùng.
        active_events (list): Danh sách các sự kiện đang hoạt động.
        interactions_df (pd.DataFrame): DataFrame chứa dữ liệu tương tác.
        user_preferences (dict): Sở thích người dùng.
        alpha (float): Trọng số cho CBF (0 <= alpha <= 1).
        top_k (int): Số lượng gợi ý tối đa.

    Returns:
        list: Danh sách eventId được gợi ý.
    """
    # Tính điểm CBF
    cbf_scores = get_cbf_scores(active_events, user_preferences)

    # Tính ma trận tương đồng cho CF
    sim_df = get_similarity_matrix(interactions_df)

    # Lấy gợi ý từ CBF
    sorted_cbf_suggestions = sorted(cbf_scores.items(), key=lambda x: x[1], reverse=True)
    cbf_suggestions = [eid for eid,  _ in sorted_cbf_suggestions]

    # Lấy gợi ý từ CF
    cf_suggestions = get_cf_recommendations(user_id,sim_df, interactions_df, top_k)

    # Kết hợp CBF và CF
    all_event_ids = set(cbf_suggestions) | set(cf_suggestions)
    hybrid_scores = {}
    for eid in all_event_ids:
        cbf = cbf_scores.get(eid, 0)
        cf = 1 if eid in cbf_suggestions else 0
        hybrid_scores[eid] = alpha * cbf + (1-alpha) * cf

    sorted_recs = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)
    return [eid for eid, _ in sorted_recs[:top_k]]

