import logging
import config
from .sentiment_analyzer import SentimentAnalyzer

logger = logging.getLogger(__name__)

def analyze_aspects(reviews: list, sentiment_analyzer: SentimentAnalyzer):
    logger.info("Đang phân tích cảm xúc theo khía cạnh...")
    aspect_sentiments = {key: [] for key in config.ASPECT_KEYWORDS.keys()}

    for review in reviews:
        for aspect, keywords in config.ASPECT_KEYWORDS.items():

            if any(keyword in review.lower() for keyword in keywords):
                result = sentiment_analyzer.analyze([review])
                if result:
                    aspect_sentiments[aspect].append(result[0]['score'])

    final_aspect_scores = []
    for aspect, scores in aspect_sentiments.items():
        if scores:
            avg_score = sum(scores) / len(scores)
            final_aspect_scores.append({
                "aspect": aspect,
                "score": avg_score,
                "sentiment": "positive" if avg_score > 0.1 else "negative" if avg_score < -0.1 else "neutral",
                "mention_count": len(scores)
            })

    logger.info("✓ Phân tích cảm xúc theo khía cạnh thành công.")
    return final_aspect_scores