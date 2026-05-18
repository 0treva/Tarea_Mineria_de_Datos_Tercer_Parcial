import re
from typing import List

try:
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer, WordNetLemmatizer
    NLTK_AVAILABLE = True
except Exception:
    NLTK_AVAILABLE = False

BASIC_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "by", "for", "from", "has",
    "have", "he", "her", "his", "i", "in", "is", "it", "its", "me", "my", "of", "on",
    "or", "our", "she", "that", "the", "their", "them", "this", "to", "was", "were", "will",
    "with", "you", "your", "very", "really", "just", "also", "but", "not"
}


def get_stopwords(language: str = "english") -> set:
    """Obtiene stopwords de NLTK. Si no están instaladas, usa una lista básica."""
    if NLTK_AVAILABLE:
        try:
            return set(stopwords.words(language))
        except Exception:
            return BASIC_STOPWORDS
    return BASIC_STOPWORDS


def clean_text(text: str) -> str:
    """Limpia texto: minúsculas, URLs, HTML, signos, números y espacios extra."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    """Tokenización simple por espacios después de limpiar el texto."""
    return clean_text(text).split()


def fallback_lemmatize(token: str) -> str:
    """Normalización simple para ambientes sin WordNet."""
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("ing") and len(token) > 5:
        return token[:-3]
    if token.endswith("ed") and len(token) > 4:
        return token[:-2]
    if token.endswith("s") and len(token) > 3:
        return token[:-1]
    return token


def preprocess_text(text: str, method: str = "lemmatization") -> str:
    """
    Aplica limpieza, tokenización, eliminación de stopwords y stemming o lemmatization.

    method:
    - stemming
    - lemmatization
    """
    tokens = tokenize(text)
    stops = get_stopwords("english")
    tokens = [token for token in tokens if token not in stops and len(token) > 1]

    if method == "stemming":
        if NLTK_AVAILABLE:
            stemmer = PorterStemmer()
            tokens = [stemmer.stem(token) for token in tokens]
        else:
            tokens = [fallback_lemmatize(token) for token in tokens]
    elif method == "lemmatization":
        if NLTK_AVAILABLE:
            try:
                lemmatizer = WordNetLemmatizer()
                tokens = [lemmatizer.lemmatize(token) for token in tokens]
            except Exception:
                tokens = [fallback_lemmatize(token) for token in tokens]
        else:
            tokens = [fallback_lemmatize(token) for token in tokens]
    else:
        raise ValueError("method debe ser 'stemming' o 'lemmatization'")

    return " ".join(tokens)
