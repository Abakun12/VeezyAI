

def get_cbf_scores(active_events, user_preferences):
    """
    Tính điểm Content-Based Filtering cho các sự kiện dựa trên sở thích người dùng.

    Args:
        active_events (list): Danh sách các sự kiện đang hoạt động.
        user_preferences (dict): Sở thích người dùng, chứa 'preferredTags' và 'preferredCategories'.

    Returns:
        dict: Dictionary với key là eventId và value là điểm CBF.
    """
    cbf_scores = {}

    for event in active_events:
        score = 0
        for tag in event['tags']:
            if tag in user_preferences['preferredTags']:
                score += 1
        if event['category'] in user_preferences['preferredCategories']:
            score += 1
        cbf_scores[event['eventId']] = score
    return cbf_scores