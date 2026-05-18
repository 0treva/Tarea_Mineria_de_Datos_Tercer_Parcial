from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


def detect_column(columns: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    """Detecta columnas aunque cambien mayúsculas, minúsculas o separadores."""
    columns = list(columns)
    lower_map = {str(c).lower(): c for c in columns}

    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    normalized = {
        str(c).lower().replace("_", "").replace(" ", ""): c
        for c in columns
    }
    for candidate in candidates:
        candidate_norm = candidate.lower().replace("_", "").replace(" ", "")
        if candidate_norm in normalized:
            return normalized[candidate_norm]

    for col in columns:
        col_norm = str(col).lower().replace("_", "").replace(" ", "")
        if any(candidate.lower().replace("_", "").replace(" ", "") in col_norm for candidate in candidates):
            return col

    return None


def load_reviews(path: str | Path) -> pd.DataFrame:
    """
    Carga reseñas y crea una etiqueta binaria de sentimiento.

    Reglas:
    - rating >= 4: sentimiento positivo, etiqueta 1.
    - rating <= 2: sentimiento negativo, etiqueta 0.
    - rating == 3: se elimina por ser neutral.

    El código acepta nombres comunes de columnas del dataset de Kindle:
    reviewText, review_text, review, text, summary, overall, rating, score y stars.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {path}")

    df = pd.read_csv(path)

    text_col = detect_column(
        df.columns,
        ["reviewText", "review_text", "review", "text", "summary", "reviewText"],
    )
    rating_col = detect_column(
        df.columns,
        ["overall", "rating", "score", "stars", "calificacion", "calificación"],
    )

    if text_col is None:
        raise ValueError("No se encontró una columna de texto. Usa reviewText, review_text, review, text o summary.")

    output = pd.DataFrame()
    output["text"] = df[text_col].fillna("").astype(str)

    if "sentiment" in [str(c).lower() for c in df.columns]:
        sentiment_col = detect_column(df.columns, ["sentiment"])
        output["label"] = pd.to_numeric(df[sentiment_col], errors="coerce").astype("Int64")
    else:
        if rating_col is None:
            raise ValueError("No se encontró columna de calificación. Usa overall, rating, score o stars.")
        ratings = pd.to_numeric(df[rating_col], errors="coerce")
        output["rating"] = ratings
        output = output.dropna(subset=["rating"])
        output = output[output["rating"] != 3]
        output["label"] = (output["rating"] >= 4).astype(int)

    output = output[output["text"].str.strip().str.len() > 0]
    output = output.dropna(subset=["label"])
    output["label"] = output["label"].astype(int)

    return output[["text", "label"]].reset_index(drop=True)
