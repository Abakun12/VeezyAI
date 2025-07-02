import logging
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

def extract_top_keywords(classified_reviews: list, top_n = 5):
    logger.info("Đang trích xuất từ khóa...")
    positive_texts = [r['text'] for r in classified_reviews if r['sentiment'] == 'POSITIVE']
    negative_texts = [r['text'] for r in classified_reviews if r['sentiment'] == 'NEGATIVE']

    def get_keywords(texts):
        if not texts: return []

        try:
            vectorizer = TfidfVectorizer(max_features=50, stop_words='english', ngram_range=(1, 2))
            tfidf_matrix = vectorizer.fit_transform(texts)

            feature_names = vectorizer.get_feature_names_out()
            return feature_names[:top_n].tolist()
        except Exception:
            words = ''.join(texts).lower().split()

            stop_words = set(["and", "the", "to", "a", "is", "in", "it", "i", "was", "for"])
            words = [word for word in words if word.isalpha() and word not in stop_words]
            return [word for word, count in Counter(words).most_common(top_n)]

    top_positive = get_keywords(positive_texts)
    top_negative = get_keywords(negative_texts)
    logger.info("✓ Trích xuất từ khóa thành công.")
    return {"positive": top_positive, "negative": top_negative}

