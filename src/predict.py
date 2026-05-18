import argparse

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .config import DATA_SAMPLE, RANDOM_STATE, TEST_SIZE
from .data_loader import load_reviews
from .preprocessing import preprocess_text


def train_default_model(data_path=DATA_SAMPLE):
    """Entrena un modelo base TF-IDF + Logistic Regression para probar frases nuevas."""
    df = load_reviews(data_path)
    df["clean_text"] = df["text"].apply(lambda value: preprocess_text(value, method="lemmatization"))

    X_train, _, y_train, _ = train_test_split(
        df["clean_text"],
        df["label"],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["label"],
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ("classifier", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ])
    model.fit(X_train, y_train)
    return model


def predict_sentiment(text: str, data_path=DATA_SAMPLE):
    model = train_default_model(data_path)
    clean = preprocess_text(text, method="lemmatization")
    prediction = model.predict([clean])[0]
    return "Positivo" if prediction == 1 else "Negativo"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clasifica una reseña nueva.")
    parser.add_argument("--data", default=str(DATA_SAMPLE), help="Ruta del CSV de entrenamiento")
    parser.add_argument("--text", required=True, help="Texto de la reseña a clasificar")
    args = parser.parse_args()

    print(predict_sentiment(args.text, args.data))
