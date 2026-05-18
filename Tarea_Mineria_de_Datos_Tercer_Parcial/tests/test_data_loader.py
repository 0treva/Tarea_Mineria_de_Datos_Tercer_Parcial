from src.config import DATA_SAMPLE
from src.data_loader import load_reviews


def test_load_reviews_sample():
    df = load_reviews(DATA_SAMPLE)
    assert "text" in df.columns
    assert "label" in df.columns
    assert set(df["label"].unique()) == {0, 1}
