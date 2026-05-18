from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RANDOM_STATE = 42
TEST_SIZE = 0.20
MAX_FEATURES = 5000
NGRAM_RANGE = (1, 2)
MIN_DF = 1

DATA_SAMPLE = PROJECT_ROOT / "data" / "sample" / "kindle_reviews_sample.csv"
DATA_RAW_DEFAULT = PROJECT_ROOT / "data" / "raw" / "kindle_reviews.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
REPORTS_DIR = OUTPUT_DIR / "reports"
