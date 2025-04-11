def clean_text(text: str) -> str:
    """Clean and normalize text."""
    if not isinstance(text, str):
        return ""
    return text.strip().replace("\r\n", "\n").replace("\t", " ")