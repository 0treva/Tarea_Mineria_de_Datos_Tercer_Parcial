from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
    precision_recall_fscore_support,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC


def build_vectorizer(representation: str, max_features: int, ngram_range=(1, 2), min_df=1):
    """Construye el vectorizador BoW o TF-IDF."""
    if representation == "bow":
        return CountVectorizer(max_features=max_features, ngram_range=ngram_range, min_df=min_df)
    if representation == "tfidf":
        return TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, min_df=min_df)
    raise ValueError("representation debe ser 'bow' o 'tfidf'")


def get_models():
    """Regresa los modelos solicitados para comparar desempeño."""
    return {
        "Naive_Bayes": MultinomialNB(),
        "Logistic_Regression": LogisticRegression(max_iter=1000, random_state=42),
        "SVM_Lineal": LinearSVC(random_state=42),
    }


def evaluate_model(model, X_train, X_test, y_train, y_test):
    """Calcula métricas de clasificación y matriz de confusión."""
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_test_pred, average="binary", zero_division=0
    )
    cm = confusion_matrix(y_test, y_test_pred, labels=[0, 1])

    test_loss = None
    if hasattr(model, "predict_proba"):
        try:
            probabilities = model.predict_proba(X_test)
            test_loss = log_loss(y_test, probabilities, labels=[0, 1])
        except Exception:
            test_loss = None

    return {
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "test_loss": test_loss,
        "confusion_matrix": cm,
        "classification_report": classification_report(
            y_test, y_test_pred, target_names=["Negativo", "Positivo"], zero_division=0
        ),
    }
