from src.chunking.strategies.text_tiling import (
    PseudosentenceBoundaryTextTilingTokenizer,
)
from src.chunking import get_chunker


def test_text_tiling_chunks_flat_text_without_paragraph_breaks():
    text = (
        ("apple banana orchard fruit harvest " * 80)
        + ("quantum electron photon particle energy " * 80)
        + ("violin symphony melody concert orchestra " * 80)
    ).strip()

    chunker = get_chunker("text_tiling", {"show_progress": False})
    chunks = chunker.split_text([text])

    assert len(chunks) > 1
    assert "".join(chunk.text for chunk in chunks).replace(" ", "") == text.replace(" ", "")


def test_pseudosentence_breaks_translate_after_removed_punctuation():
    tokenizer = PseudosentenceBoundaryTextTilingTokenizer(
        w=3,
        k=1,
        stopwords=[],
        smoothing_width=2,
        smoothing_rounds=1,
    )
    text = "Alpha beta gamma. Delta epsilon zeta."
    nopunct_text, offset_map = tokenizer._remove_punctuation_with_map(text.lower())
    tokseqs = tokenizer._divide_to_tokensequences(nopunct_text)

    nopunct_breaks = tokenizer._mark_pseudosentence_breaks(tokseqs)
    original_breaks = tokenizer._translate_breaks_to_original_text(
        nopunct_breaks,
        offset_map,
        text,
    )

    assert text[original_breaks[1] - 1] == "."
    assert text[: original_breaks[1]] == "Alpha beta gamma."


def test_stopwords_language_resolves_nltk_language(monkeypatch):
    requested_languages = []

    def fake_words(language):
        requested_languages.append(language)
        return ["i", "oraz"]

    monkeypatch.setattr(
        "src.chunking.strategies.text_tiling._load_nltk_stopwords",
        fake_words,
    )

    chunker = get_chunker(
        "text_tiling",
        {
            "stopwords": None,
            "stopwords_language": "polish",
            "show_progress": False,
        },
    )

    assert requested_languages == ["polish"]
    assert chunker.tokenizer.stopwords == ["i", "oraz"]


def test_polish_stopwords_fall_back_to_spacy(monkeypatch):
    monkeypatch.setattr(
        "src.chunking.strategies.text_tiling._load_nltk_stopwords",
        lambda language: (_ for _ in ()).throw(OSError("missing language")),
    )
    monkeypatch.setattr(
        "src.chunking.strategies.text_tiling._load_polish_stopwords",
        lambda: ["i", "oraz"],
    )

    chunker = get_chunker(
        "text_tiling",
        {
            "stopwords": None,
            "stopwords_language": "polish",
            "show_progress": False,
        },
    )

    assert chunker.tokenizer.stopwords == ["i", "oraz"]
