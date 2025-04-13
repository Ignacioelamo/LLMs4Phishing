# utils.py
import pandas as pd
import spacy
from typing import Union, List

def preprocess_texts(
    texts: Union[List[str], pd.Series],
    spacy_model: str = "en_core_web_sm",
    batch_size: int = 200,
    n_process: int = 8
) -> List[str]:
    """
    Preprocess a list or pandas Series of texts by lowercasing, removing stopwords and punctuation,
    and lemmatizing using spaCy.

    Args:
        texts (Union[List[str], pd.Series]): Input texts to preprocess.
        spacy_model (str): Name of the spaCy model to load (default: "en_core_web_sm").
        batch_size (int): Batch size for spaCy processing (default: 200).
        n_process (int): Number of processes for parallel spaCy processing (default: 8).

    Returns:
        List[str]: List of cleaned texts.
    """
    # Load spaCy model
    try:
        nlp = spacy.load(spacy_model, disable=["ner", "parser"])
    except OSError:
        raise ValueError(f"spaCy model '{spacy_model}' not found. Install it with: python -m spacy download {spacy_model}")

    # Convert input to list and handle NaN/None values
    if isinstance(texts, pd.Series):
        texts = texts.fillna("").astype(str).tolist()
    else:
        texts = [str(text) if pd.notna(text) else "" for text in texts]

    # Lowercase texts
    texts = [text.lower() for text in texts]

    # Process texts in parallel with spaCy
    cleaned_texts = []
    for doc in nlp.pipe(texts, batch_size=batch_size, n_process=n_process):
        tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
        cleaned_texts.append(" ".join(tokens))

    return cleaned_texts