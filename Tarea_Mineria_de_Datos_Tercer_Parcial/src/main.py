import argparse
import time
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .config import (
    DATA_SAMPLE,
    FIGURES_DIR,
    MAX_FEATURES,
    MIN_DF,
    NGRAM_RANGE,
    OUTPUT_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
    TEST_SIZE,
)
from .data_loader import load_reviews
from .evaluate import plot_confusion_matrix, plot_results_bar
from .modeling import build_vectorizer, evaluate_model, get_models
from .preprocessing import preprocess_text


def run_experiment(data_path: str | Path = DATA_SAMPLE, limit: int | None = None) -> pd.DataFrame:
    """Ejecuta todos los experimentos del mini proyecto."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_reviews(data_path)
    if limit is not None and limit > 0:
        df = df.sample(min(limit, len(df)), random_state=RANDOM_STATE)

    if df["label"].nunique() < 2:
        raise ValueError("Se necesitan reseñas positivas y negativas para entrenar.")

    print(f"Dataset cargado: {len(df)} reseñas")
    print(df["label"].value_counts().rename(index={0: "Negativo", 1: "Positivo"}))

    results = []
    reports_text = []
    representations = ["bow", "tfidf"]
    processing_methods = ["stemming", "lemmatization"]

    for method in processing_methods:
        print(f"\nPreprocesamiento: {method}")
        df[f"text_{method}"] = df["text"].apply(lambda value: preprocess_text(value, method=method))

        X_train, X_test, y_train, y_test = train_test_split(
            df[f"text_{method}"],
            df["label"],
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=df["label"],
        )

        for representation in representations:
            print(f"Representación: {representation.upper()}")

            for model_name, estimator in get_models().items():
                vectorizer = build_vectorizer(representation, MAX_FEATURES, NGRAM_RANGE, MIN_DF)
                pipeline = Pipeline([
                    ("vectorizer", vectorizer),
                    ("classifier", estimator),
                ])

                start = time.time()
                pipeline.fit(X_train, y_train)
                elapsed = time.time() - start

                metrics = evaluate_model(pipeline, X_train, X_test, y_train, y_test)
                config_name = f"{representation.upper()} + {method}"

                row = {
                    "configuracion": config_name,
                    "representacion": representation,
                    "procesamiento": method,
                    "modelo": model_name,
                    "train_accuracy": round(metrics["train_accuracy"], 4),
                    "test_accuracy": round(metrics["test_accuracy"], 4),
                    "precision": round(metrics["precision"], 4),
                    "recall": round(metrics["recall"], 4),
                    "f1_score": round(metrics["f1_score"], 4),
                    "test_loss": None if metrics["test_loss"] is None else round(metrics["test_loss"], 4),
                    "tiempo_entrenamiento_seg": round(elapsed, 4),
                    "num_documentos": len(df),
                }
                results.append(row)

                safe_name = f"{representation}_{method}_{model_name}"
                fig_path = FIGURES_DIR / f"confusion_matrix_{safe_name}.png"
                plot_confusion_matrix(metrics["confusion_matrix"], f"{model_name} - {config_name}", fig_path)

                reports_text.append("=" * 80)
                reports_text.append(f"Modelo: {model_name}")
                reports_text.append(f"Configuración: {config_name}")
                reports_text.append(str(row))
                reports_text.append(metrics["classification_report"])

                print(
                    f"  {model_name}: "
                    f"Train Acc={row['train_accuracy']} | "
                    f"Test Acc={row['test_accuracy']} | "
                    f"F1={row['f1_score']} | "
                    f"Tiempo={row['tiempo_entrenamiento_seg']}s"
                )

    results_df = pd.DataFrame(results).sort_values(by="f1_score", ascending=False)
    results_path = REPORTS_DIR / "resultados_comparativos.csv"
    reports_path = REPORTS_DIR / "classification_reports.txt"

    results_df.to_csv(results_path, index=False)
    reports_path.write_text("\n".join(reports_text), encoding="utf-8")
    plot_results_bar(results_df, FIGURES_DIR / "comparacion_f1_score.png")

    print("\nResultados guardados en:")
    print(results_path)
    print(reports_path)
    print("\nMejor configuración:")
    print(results_df.head(1).to_string(index=False))

    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mini proyecto de Sentiment Analysis con NLP")
    parser.add_argument("--data", type=str, default=str(DATA_SAMPLE), help="Ruta del CSV de reseñas")
    parser.add_argument("--limit", type=int, default=None, help="Límite opcional de filas para pruebas rápidas")
    args = parser.parse_args()

    run_experiment(args.data, args.limit)
