import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
import logging
import config

logger = logging.getLogger(__name__)


def train_cf_model(interaction_df):
    """Huấn luyện mô hình Lọc Cộng tác (SVD) từ ma trận tương tác."""
    logger.info("Đang huấn luyện mô hình Lọc Cộng tác (TruncatedSVD)...")
    if interaction_df.empty:
        logger.warning("Không có dữ liệu tương tác để huấn luyện CF.")
        return None

    # Tạo ma trận tương tác: user-item
    interaction_matrix = interaction_df.pivot_table(
        index='accountId',
        columns='eventId',
        aggfunc='size',
        fill_value=0
    ).applymap(lambda x: 1 if x > 0 else 0)

    if interaction_matrix.empty or interaction_matrix.shape[1] < 2:
        logger.warning("Không đủ dữ liệu để tạo ma trận tương tác.")
        return None

    # Giảm chiều dữ liệu bằng TruncatedSVD
    n_components = min(config.CF_N_COMPONENTS, interaction_matrix.shape[1] - 1)
    if n_components < 1:
        logger.warning(f"Không đủ sự kiện ({interaction_matrix.shape[1]}) để huấn luyện SVD.")
        return None

    svd = TruncatedSVD(n_components=n_components, random_state=42)
    user_features_matrix = svd.fit_transform(interaction_matrix.values)

    logger.info("✓ Huấn luyện mô hình Lọc Cộng tác thành công.")
    return {
        "model": svd,
        "interaction_matrix": interaction_matrix,
        "user_features": user_features_matrix,
    }

def get_cf_recommendations(account_id, cf_model_info, events_user_interacted, top_k=10):
    """Lấy gợi ý từ mô hình Lọc Cộng tác đã huấn luyện."""
    if cf_model_info is None:
        return {}

    model = cf_model_info.get('model')
    interaction_matrix = cf_model_info.get('interaction_matrix')
    user_features = cf_model_info.get('user_features')

    if model is None or interaction_matrix is None:
        return {}

    try:
        user_index = interaction_matrix.index.get_loc(account_id)
    except KeyError:
        # Người dùng không có trong ma trận tương tác
        return {}

    # Dự đoán điểm cho tất cả các item
    user_latent_vector = user_features[user_index]
    predicted_scores = np.dot(user_latent_vector, model.components_)

    # Tạo series, loại bỏ các item đã tương tác và lấy top K
    cf_series = pd.Series(predicted_scores, index=interaction_matrix.columns).drop(events_user_interacted,
                                                                                   errors='ignore')
    return cf_series.sort_values(ascending=False).head(top_k).to_dict()
