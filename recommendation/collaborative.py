import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
import logging
import config

logger = logging.getLogger(__name__)

def train_cf_model_sklearn(feedback_df):
    logger.info("Đang huấn luyện mô hình Lọc Cộng tác (TruncatedSVD)...")
    if feedback_df.empty:
        logger.warning("Không có dữ liệu feedback để huấn luyện CF.")
        return None
    utility_matrix = feedback_df.pivot_table(index='accountId', columns='eventId', values='rating').fillna(0)
    if utility_matrix.empty:
        logger.warning("Không thể tạo ma trận tương tác.")
        return None
    n_components = min(config.CF_N_COMPONENTS, utility_matrix.shape[1]- 1)
    if n_components < 1:
        logger.warning(f"Không đủ sự kiện ({utility_matrix.shape[1]}) để huấn luyện mô hình SVD.")
        return None
    svd = TruncatedSVD(n_components=n_components , random_state=42)
    user_features_matrix = svd.fit_transform(utility_matrix.values)
    logger.info("✓ Huấn luyện mô hình Lọc Cộng tác thành công.")
    return {
        "mode":svd,
        "utility_matrix": utility_matrix,
        "user_features": user_features_matrix,
    }

def get_cf_recommendations_sklearn(account_id, cf_model_info, events_user_interacted, top_k=10):
    cf_model = cf_model_info.get['mode']
    utility_matrix = cf_model_info.get['utility_matrix']
    user_features_matrix  = cf_model_info.get['user_features']
    if cf_model is None or utility_matrix is None: return {}
    try:
        user_index = utility_matrix.index.get_loc(account_id)
    except KeyError:
        return {}
    user_latent_vector = user_features_matrix[user_index]
    predicted_ratings = np.dot(user_latent_vector, cf_model.components_)
    cf_series = pd.Series(predicted_ratings, index=utility_matrix.columns).drop(events_user_interacted, errors='ignore')
    top_recommendations = cf_series.sort_values(ascending=False).head(top_k)
    return top_recommendations.to_dict()