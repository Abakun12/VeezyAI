from transformers import pipeline
import logging
import config

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    def __init__(self):
        logger.info(f"Đang tải mô hình phân tích cảm xúc: {config.SENTIMENT_MODEL_NAME}...")
        try:
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=config.SENTIMENT_MODEL_NAME
            )
            logger.info(f"✓ Mô hình phân tích cảm xúc đã sẵn sàng.")
        except Exception as e:
            logger.exception(f"Lỗi nghiêm trọng khi tải mô hình sentiment: {e}")
            self.sentiment_pipeline = None

    def analyze (self, text_list: list):
        if not self.sentiment_pipeline:
            logger.warning("Mô hình sentiment chưa được tải, không thể phân tích.")
            return []

        try:
            results =self.sentiment_pipeline(text_list)

            processed_results = []
            for result in results:
                score = result['score'] if result['label'] == 'POSITIVE' else -result['score']
                processed_results.append({'label': result['label'], 'score': score})
            return processed_results
        except Exception as e:
            logger.exception(f"Lỗi khi phân tích cảm xúc: {e}")
            return []