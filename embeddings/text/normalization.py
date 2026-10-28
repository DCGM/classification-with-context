"""The page text as the page-text encoders get it, free of layout.

The text dump keeps the PAGE XML lines as they are: blank ones included, the page capped at 4000 characters.
Embeddings computed from it as it is learn cues that exist only in the dump (a page with little text shows
them most), and line breaks are layout anyway: line counts, lengths and positions are what the page statistics
are for, and mmBERT, pretrained on continuous text, is not known to read a line break sensibly. So the text
is reduced to its words, separated by single spaces, in the OCR reading order; the cap is applied after
that. MetaKat has to build the same string (metakat.page_type.engines.core.context.mmbert / alto_stats).
"""
from typing import Callable, Dict, Optional

MAX_CHARS = 4000


def normalize_ocr_words(text: str, max_chars: Optional[int] = MAX_CHARS) -> str:
    """All words of the page, one space between them, then the first max_chars characters; '' for a blank page."""
    text = ' '.join(text.split())
    return text if max_chars is None else text[:max_chars]


def normalize_ocr_lines(text: str, max_chars: Optional[int] = MAX_CHARS) -> str:
    """Lines kept (the engine's text of v1.7.0): one space between words, no blank lines, then the first max_chars characters; '' for a blank page."""
    lines = (' '.join(line.split()) for line in text.split('\n'))
    text = '\n'.join(line for line in lines if line)
    return text if max_chars is None else text[:max_chars]


def keep_text(text: str, max_chars: Optional[int] = None) -> str:
    """The dump text unchanged, as the first mmBERT and Qwen3 embeddings were computed from it."""
    return text


NORMALIZATIONS: Dict[str, Callable[..., str]] = {'ocr_words': normalize_ocr_words, 'ocr_lines': normalize_ocr_lines, 'none': keep_text}
