import os

def load_text_file(filepath: str) -> str:
    """
    Reads a text file and returns its content as a string.
    Raises an error if file is not found.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()