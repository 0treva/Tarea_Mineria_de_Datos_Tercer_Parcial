from src.preprocessing import clean_text, preprocess_text


def test_clean_text_removes_numbers_and_punctuation():
    text = "Hello!!! This book is good in 2026."
    cleaned = clean_text(text)
    assert "2026" not in cleaned
    assert "!" not in cleaned
    assert cleaned == cleaned.lower()


def test_preprocess_text_returns_string():
    result = preprocess_text("This book was running very well.", method="stemming")
    assert isinstance(result, str)
    assert len(result) > 0
