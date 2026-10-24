"""Page statistics "model": a fixed, non-neural embedding of a page from its OCR (ALTO; plain text as a
fallback) made of keyword hits, regex patterns, line statistics and ALTO line geometry. See README.md
for the features, the keyword groups and the design decisions.

The full vector (FEATURES) is what gets stored (e.g. in an LMDB, as float16): keyword hits per stem as
raw counts, so the stems can be regrouped, dropped or log-scaled later. PageStats.view() turns stored
full vectors into the vectors a model uses: per stem or grouped (GROUPED_FEATURES, the vector of the
experiments), with selected features or blocks only.

Only numpy and the standard library; nothing here prints or logs page text, so it can run on private
pages inside a script that writes only vectors or predictions.

    from embeddings.text.page_stats import PageStats
    stats = PageStats()
    full = stats.embed_alto(alto_bytes)                 # len(FEATURES) numbers, float32
    full = stats.embed([alto1, alto2], workers=8)       # matrix, one row per page
    x, names = stats.view(full, grouped=True, exclude=['alto_word_conf', 'alto_word_boxes'])

    python -m embeddings.text.page_stats.page_stats page.xml --grouped   # nonzero features of one page
"""
import argparse
import fnmatch
import re
import unicodedata
import xml.etree.ElementTree as ET
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

VERSION = '1'

# ---------------------------------------------------------------------------------------------------------
# Keywords: group -> stems (lower case, no diacritics), matched at the start of a word, so a stem covers
# its inflected forms ('predmluv': předmluva, předmluvy, předmluvou). A stem with spaces is a phrase; a
# trailing space ends the word ('viz '). A stem may be in several groups; it is counted once per page.
# ---------------------------------------------------------------------------------------------------------
KEYWORD_GROUPS: Dict[str, Tuple[str, ...]] = {
    'toc': ('obsah', 'inhalt', 'contents', 'table des matieres', 'sommaire', 'spis tresci', 'tartalom',
            'содержание', 'оглавление', 'index capitum', 'indice'),
    'index': ('rejstrik', 'registrik', 'rejstr', 'register', 'index', 'sachregister', 'namenregister',
              'jmenny', 'vecny', 'ukazatel', 'указатель', 'viz '),
    'errata': ('errata', 'corrigenda', 'oprav', 'tiskov', 'druckfehler', 'berichtigung', 'rettelser', 'omyl',
               'radek', 'shora', 'zdola', 'misto ', 'cti ', 'lies ', 'statt ', 'zeile', 'additions'),
    'preface': ('predmluv', 'predml', 'vorrede', 'vorwort', 'preface', 'praefatio', 'prefatio',
                'avant-propos', 'avant propos', 'uvodem', 'uvod', 'einleitung', 'introduction', 'предислов'),
    'dedication': ('venovan', 'venuj', 'pamatce', 'na pamet', 'dedicatoria', 'dedicat', 'illustrissim',
                   'reverendissim', 'excellentissim', 'gewidmet', 'zugeeignet', 'dedie', 'dedicated', 'meinem',
                   'meiner', 'drahe', 'milovan'),
    'imprimatur': ('imprimatur', 'nihil obstat', 'approbat', 'privileg', 'censor', 'permis', 'licentia',
                   'facultas', 'admittitur', 'typis ', 'vicarius', 'ordinariat'),
    'imprint': ('isbn', 'issn', 'vytiskl', 'tiskem', 'tiskarn', 'knihtiskarn', 'vytisk', 'vydal', 'nakladem',
                'naklad ', 'nakladatel', 'vydani', 'odpovedny redaktor', 'redaktor', 'sazb', 'autorskych arch',
                'autorske arch', 'prava vyhrazena', 'vsechna prava', 'copyright', 'printed in', 'gedruckt',
                'druck von', 'druck und verlag', 'verlag', 'imprime', 'brozovan', 'vazan', 'tematicka skupina',
                'povoleno', 'schvalil', 'edice', 'svazek'),
    'bibliography': ('bibliografi', 'literatura', 'literatur', 'prameny', 'pramen', 'seznam literatury',
                     'pouzita literatura', 'citovan', 'literaturverzeichnis', 'quellen', 'bibliography',
                     'references', 'bibliographie', 'библиограф', 'литература'),
    'list_illustrations': ('seznam vyobrazeni', 'seznam ilustraci', 'seznam obrazk', 'seznam obrazu',
                           'vyobrazeni', 'ilustrac', 'obrazk', 'obr ', 'abbildung', 'illustrations', 'figures',
                           'fig ', 'tafel', 'planche', 'figur'),
    'list_maps': ('seznam map', 'mapy', 'map ', 'maps', 'karten', 'cartes', 'plans', 'plany', 'planu'),
    'list_tables': ('seznam tabulek', 'tabulek', 'tabulky', 'tabellen', 'tables', 'tableaux', 'prehled tabul'),
    'appendix': ('priloh', 'dodatek', 'dodatk', 'appendix', 'anhang', 'beilage', 'supplement', 'nachtrag'),
    'abstract': ('abstract', 'abstrakt', 'summary', 'resume', 'zusammenfassung', 'souhrn', 'keywords',
                 'klicova slova', 'klicova', 'резюме', 'аннотац'),
    'advertisement': ('cena ', 'k dostani', 'prodej', 'prodava', 'nabizi', 'objednav', 'adresa', 'telefon',
                      'tel ', 'inzer', 'inserat', 'anzeige', 'verkauf', 'zu haben', 'preis', 'firma', 'sklad',
                      'zdarma', 'levne', 'vermiet', 'zu verkaufen', 'empfiehlt', 'doporucuje', 'zavod'),
    'title': ('napsal', 'sepsal', 'prelozil', 'usporadal', 'upravil', 'vydava', 'von ', 'herausgegeben',
              'ubersetzt', 'par ', 'auctore', 'editio', 'dil ', 'band ', 'tome', 'volume', 'praha', 'v praze',
              'wien', 'leipzig', 'brno', 'prag'),
    'masthead': ('rocnik', 'cislo', 'jahrgang', 'nummer', 'predplatn', 'vychazi', 'administrac',
                 'zeitung', 'blatt', 'abonnement', 'bezugspreis', 'erscheint', 'redakce'),
    'calibration': ('colorchecker', 'xrite', 'x-rite', 'kodak', 'color control', 'gray scale', 'grey scale',
                    'ektacolor', 'target', 'cyan', 'magenta', 'yellow', 'colour', 'golden thread'),
    'music': ('allegro', 'andante', 'moderato', 'adagio', 'presto', 'largo', 'lento', 'vivace', 'piano', 'pianoforte',
              'forte', 'cresc', 'cres ', 'dim ', 'decresc', 'rit ', 'rall', 'pizz', 'arco', 'fine ', 'coda',
              'tempo', 'dolce', 'maestoso', 'andantino', 'allegretto', 'violin', 'soprano', 'alto ', 'tenor',
              'basso', 'singstimme', 'klavir', 'zpev'),
    'map': ('meritko', 'mapa', 'karte', 'carte', 'map ', 'massstab', 'masstab', 'scale', 'vrstevnic',
            'legenda', 'zeichenerklarung', 'souradnic', 'poledni', 'severni', 'km ', 'kilometr', 'statute miles'),
    'stamp': ('knihovna', 'knihovn', 'inv ', 'inventar', 'signatura', 'znac ', 'bibliothek', 'library',
              'bibliotheca', 'ex libris', 'exlibris', 'majetek', 'razitko'),
    'table': ('tab ', 'tabulka', 'tabelle', 'celkem', 'summa', 'prumer', 'soucet', 'zusammen', 'total'),
    'cover': ('cena ', 'brozovan', 'vazan', 'edice', 'knihovna', 'svazek', 'sesit', 'heft', 'lieferung'),
    'end': ('konec', 'ende ', 'finis', 'the end', 'fin ', 'explicit', 'tiskem', 'vytiskl', 'vytisk'),
}
STEMS: Tuple[str, ...] = tuple(dict.fromkeys(s for stems in KEYWORD_GROUPS.values() for s in stems))

# regex patterns, matched per line on the lower-cased text without diacritics (punctuation kept)
PATTERNS: Dict[str, re.Pattern] = {
    'isbn': re.compile(r'\bisbn\b|\bissn\b|\b\d{2,5}-\d{2,7}-\d{2,7}-[\dx]\b'),
    'price': re.compile(r'\b\d+(?:[,.]\d+)?\s*(?:kc|kcs|k|h|kr|zl|mk|fr|fl|rm|dm|eur|sk)\b\.?|\bcena\b'),
    'year': re.compile(r'(?<!\d)(?:1[5-9]\d\d|20[0-2]\d)(?!\d)'),
    'roman_year': re.compile(r'\bm\s*d?c{0,4}[lx]{0,4}[vi]{0,4}\b\.?$'),
    'scale': re.compile(r'1\s*:\s*\d[\d\s.]{2,}'),
    'page_ref': re.compile(r'\b(?:str|s|seite|p|pag)\.?\s*\d+'),
    'leader_number': re.compile(r'(?:\.{3,}|(?:\. ){3,}|…)\s*\d+\s*$'),
    'figure_ref': re.compile(r'\b(?:obr|fig|abb|tab)\.?\s*\d+'),
}
EDGE_LINES = 3  # header = first 3, footer = last 3 non-empty lines; a page of up to 3 lines has no footer

TEXT_LINE_FEATURES = ('log_chars', 'log_lines', 'empty', 'mean_line_len', 'std_line_len', 'short_lines',
                      'lines_end_number', 'number_only_lines', 'index_lines', 'dot_leader_lines', 'list_start_lines',
                      'digit_ratio', 'upper_ratio', 'allcaps_lines', 'punct_ratio', 'alpha_ratio', 'words_per_line',
                      'alphabetical_lines', 'numeric_lines', 'multi_number_lines', 'first_line_caps')
ALTO_FEATURES = (
    'page_aspect', 'n_blocks', 'n_lines', 'n_words',
    'text_area', 'text_top', 'text_bottom', 'text_left', 'text_right', 'text_center_x', 'text_center_y',
    'line_height_med', 'line_height_max_ratio', 'large_lines', 'top_line_ratio', 'top_line_centered',
    'centered_lines', 'right_aligned_lines', 'indented_lines', 'full_width_lines', 'columns',
    'line_gap_med', 'line_gap_max', 'right_numbers', 'conf_mean', 'conf_low', 'short_words',
    'top_band_lines', 'bottom_band_lines', 'has_ocr_xml', 'word_gap_med', 'wide_gap_lines', 'char_width_ratio')
ALTO_WORD_CONF = ('conf_mean', 'conf_low')
ALTO_WORD_BOXES = ('word_gap_med', 'wide_gap_lines', 'char_width_ratio')

# blocks of the full vector: name -> feature names. Counts are raw (view() log-scales them by default).
BLOCKS: Dict[str, Tuple[str, ...]] = {
    'keywords': tuple(f'kw:{s}' for s in STEMS),            # hits on the whole page
    'head_keywords': tuple(f'head_kw:{s}' for s in STEMS),  # hits in the header lines
    'foot_keywords': tuple(f'foot_kw:{s}' for s in STEMS),  # hits in the footer lines
    'patterns': tuple(f'pat:{p}' for p in PATTERNS),         # lines matching
    'foot_patterns': tuple(f'foot_pat:{p}' for p in PATTERNS),
    'text_lines': TEXT_LINE_FEATURES,
    'alto_lines': tuple(f for f in ALTO_FEATURES if f not in ALTO_WORD_CONF + ALTO_WORD_BOXES),
    'alto_word_conf': ALTO_WORD_CONF,
    'alto_word_boxes': ALTO_WORD_BOXES,
}
COUNT_BLOCKS = ('keywords', 'head_keywords', 'foot_keywords', 'patterns', 'foot_patterns')
FEATURES: Tuple[str, ...] = (BLOCKS['keywords'] + BLOCKS['head_keywords'] + BLOCKS['foot_keywords']
                             + BLOCKS['patterns'] + BLOCKS['foot_patterns'] + TEXT_LINE_FEATURES + ALTO_FEATURES)
COUNT_FEATURES = frozenset(n for b in COUNT_BLOCKS for n in BLOCKS[b])
# the grouped vector (the one of the experiments): log(1 + hits) per group, header/footer flags per group,
# log(1 + matching lines) and footer flags per pattern, then the line and ALTO features unchanged
GROUPED_BLOCKS: Dict[str, Tuple[str, ...]] = {
    'keywords': tuple(f'kw_{g}' for g in KEYWORD_GROUPS),
    'head_keywords': tuple(f'head_{g}' for g in KEYWORD_GROUPS),
    'foot_keywords': tuple(f'foot_{g}' for g in KEYWORD_GROUPS),
    'patterns': tuple(f'pat_{p}' for p in PATTERNS),
    'foot_patterns': tuple(f'foot_pat_{p}' for p in PATTERNS),
    **{b: BLOCKS[b] for b in ('text_lines', 'alto_lines', 'alto_word_conf', 'alto_word_boxes')},
}
GROUPED_FEATURES: Tuple[str, ...] = (GROUPED_BLOCKS['keywords'] + GROUPED_BLOCKS['head_keywords']
                                     + GROUPED_BLOCKS['foot_keywords'] + GROUPED_BLOCKS['patterns']
                                     + GROUPED_BLOCKS['foot_patterns'] + TEXT_LINE_FEATURES + ALTO_FEATURES)

_STEM_PATTERNS = [re.compile(r'(?<![^\W\d_])' + re.escape(s)) for s in STEMS]
LINE_END_NUMBER = re.compile(r'\d+\s*$')
INDEX_LINE = re.compile(r'^\D{2,},\s*\d')
DOT_LEADER = re.compile(r'\.{3,}|(?:\. ){3,}|…')
LIST_START = re.compile(r'^(?:\d+[.)]|[-–•*])\s')
NUMBER_ONLY = re.compile(r'^[\s.,:;—–-]*\d[\d\s.,:;—–-]*$|^[IVXLCivxlc]+\.?$')  # a page number or roman numeral
NUMBER = re.compile(r'^[\divxlcIVXLC]+[.,]?$')


# ---------------------------------------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------------------------------------
def fold(text: str) -> str:
    """Lower case without diacritics, punctuation kept."""
    text = unicodedata.normalize('NFKD', text.lower().replace('ß', 'ss'))
    return ''.join(c for c in text if not unicodedata.combining(c))


def normalize(text: str) -> str:
    """fold() with punctuation inside lines turned into spaces (so 'Č. inv.:' -> 'c inv '), for keywords."""
    return re.sub(r'[^\w\n]+', ' ', fold(text))


def stem_counts(text: str) -> List[int]:
    return [len(p.findall(text)) for p in _STEM_PATTERNS]


def text_line_features(text: str) -> List[float]:
    """Line-structure statistics of a page text (one OCR line per text line, top to bottom)."""
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    if not lines:
        return [0.0, 0.0, 1.0] + [0.0] * (len(TEXT_LINE_FEATURES) - 3)
    chars = [c for c in text if not c.isspace()]
    n_chars = len(chars)
    lengths = np.array([len(line) for line in lines], dtype=np.float32)
    letters = [c for c in chars if c.isalpha()]
    n_letters = max(len(letters), 1)
    allcaps = [line for line in lines if sum(c.isalpha() for c in line) >= 3 and line.upper() == line]

    def share(pattern):
        return sum(1 for line in lines if pattern.search(line)) / len(lines)

    firsts = [line[0].lower() for line in lines if line[0].isalpha()]
    ordered = sum(a <= b for a, b in zip(firsts, firsts[1:])) / max(len(firsts) - 1, 1) if len(firsts) > 2 else 0.0
    digits = [sum(c.isdigit() for c in line) / max(len(line.replace(' ', '')), 1) for line in lines]
    numbers = [len(re.findall(r'\d+', line)) for line in lines]
    return [
        float(np.log1p(n_chars)), float(np.log1p(len(lines))), 0.0,
        float(lengths.mean()), float(lengths.std()), float((lengths < 20).mean()),
        share(LINE_END_NUMBER), share(NUMBER_ONLY), share(INDEX_LINE), share(DOT_LEADER), share(LIST_START),
        sum(c.isdigit() for c in chars) / max(n_chars, 1),
        sum(c.isupper() for c in letters) / n_letters,
        len(allcaps) / len(lines),
        sum(not c.isalnum() for c in chars) / max(n_chars, 1),
        len(letters) / max(n_chars, 1),
        float(np.mean([len(line.split()) for line in lines])),
        ordered, float(np.mean([d >= 0.5 for d in digits])), float(np.mean([n >= 3 for n in numbers])),
        float(sum(c.isalpha() for c in lines[0]) >= 3 and lines[0].upper() == lines[0]),
    ]


def text_vector(text: str) -> List[float]:
    """Keyword, pattern and line features of a page text (one line per text line, top to bottom)."""
    folded_lines = [line.strip() for line in fold(text).split('\n') if line.strip()]
    norm_lines = [line.strip() for line in normalize(text).split('\n') if line.strip()]
    foot_lines = EDGE_LINES if len(norm_lines) > EDGE_LINES else 0
    page = ' ' + ' '.join(norm_lines) + ' '  # the trailing space lets 'viz ' match at a line end
    head = ' ' + ' '.join(norm_lines[:EDGE_LINES]) + ' '
    foot = ' ' + ' '.join(norm_lines[len(norm_lines) - foot_lines:]) + ' ' if foot_lines else ' '
    foot_folded = folded_lines[len(folded_lines) - foot_lines:] if foot_lines else []
    patterns = [sum(1 for line in folded_lines if p.search(line)) for p in PATTERNS.values()]
    foot_patterns = [sum(1 for line in foot_folded if p.search(line)) for p in PATTERNS.values()]
    return (stem_counts(page) + stem_counts(head) + stem_counts(foot) + patterns + foot_patterns
            + text_line_features(text))


# ---------------------------------------------------------------------------------------------------------
# ALTO
# ---------------------------------------------------------------------------------------------------------
def _local(tag: str) -> str:
    return tag.rsplit('}', 1)[-1]


def parse_alto(alto: Union[bytes, str]):
    """(page width, page height, number of text blocks, lines) of an ALTO page; a line is
    (left, top, width, height, words), a word (content, confidence, left, width). None without a Page."""
    root = ET.fromstring(alto)
    page = next((e for e in root.iter() if _local(e.tag) == 'Page'), None)
    if page is None:
        return None
    width, height = float(page.get('WIDTH', 0) or 0), float(page.get('HEIGHT', 0) or 0)
    blocks, lines = 0, []
    for element in page.iter():
        tag = _local(element.tag)
        if tag == 'TextBlock':
            blocks += 1
        elif tag == 'TextLine':
            words = [(s.get('CONTENT', ''), float(s.get('WC', 1) or 1),
                      float(s.get('HPOS', 0) or 0), float(s.get('WIDTH', 0) or 0))
                     for s in element if _local(s.tag) == 'String']
            box = [float(element.get(k, 0) or 0) for k in ('HPOS', 'VPOS', 'WIDTH', 'HEIGHT')]
            if box[2] > 0 and box[3] > 0:
                lines.append((*box, words))
    return width, height, blocks, lines


def alto_text(parsed) -> str:
    """The page text from parsed ALTO: the line texts sorted top to bottom, one per line."""
    if parsed is None:
        return ''
    return '\n'.join(' '.join(word[0] for word in line[4]) for line in sorted(parsed[3], key=lambda line: line[1]))


def columns(lefts: np.ndarray, text_width: float) -> int:
    """Number of clusters of line left edges at least a quarter of the text width apart."""
    if len(lefts) < 3:
        return 1 if len(lefts) else 0
    edges = np.sort(lefts)
    gaps = np.diff(edges) > 0.25 * text_width
    starts = np.split(edges, np.flatnonzero(gaps) + 1)
    return int(sum(len(s) >= max(3, 0.1 * len(edges)) for s in starts)) or 1


def word_boxes(lines, med_h: float) -> List[float]:
    """word_gap_med, wide_gap_lines, char_width_ratio from the word boxes (zeros without word boxes)."""
    gaps, wide, char_widths, with_boxes = [], [], [], 0
    for line in lines:
        boxes = [(w[2], w[3], len(w[0])) for w in line[4] if w[2] is not None and w[3]]
        if not boxes:
            continue
        with_boxes += 1
        boxes.sort()
        line_gaps = [max(b[0] - (a[0] + a[1]), 0.0) / med_h for a, b in zip(boxes, boxes[1:])]
        gaps += line_gaps
        wide.append(any(g > 2 for g in line_gaps))
        char_widths += [w / max(n, 1) / med_h for _, w, n in boxes]
    if not with_boxes:
        return [0.0, 0.0, 0.0]
    return [float(np.median(gaps)) if gaps else 0.0, float(np.mean(wide)), float(np.median(char_widths))]


def alto_features(parsed) -> List[float]:
    """The ALTO_FEATURES of parsed ALTO (zeros without a page, has_ocr_xml = 0)."""
    zeros = [0.0] * len(ALTO_FEATURES)
    if parsed is None or parsed[0] <= 0 or parsed[1] <= 0:
        return zeros
    width, height, blocks, lines = parsed
    if not lines:
        v = zeros.copy()
        v[0], v[1], v[ALTO_FEATURES.index('has_ocr_xml')] = width / height, float(np.log1p(blocks)), 1.0
        return v
    box = np.array([line[:4] for line in lines])
    x0, y0, w, h = box.T
    x1, y1 = x0 + w, y0 + h
    left, top, right, bottom = x0.min(), y0.min(), x1.max(), y1.max()
    text_w, text_h = max(right - left, 1.0), max(bottom - top, 1.0)
    med_h = float(np.median(h))
    center = (x0 + x1) / 2
    order = np.argsort(y0)
    gaps = np.clip(y0[order][1:] - y1[order][:-1], 0, None) / med_h if len(lines) > 1 else np.zeros(1)
    first = order[0]
    centred = (np.abs(center - (left + right) / 2) < 0.05 * text_w) & (w < 0.8 * text_w)
    words = [word for line in lines for word in line[4]]
    conf = np.array([word[1] for word in words]) if words else np.ones(1)
    right_numbers = [bool(line[4]) and NUMBER.match(line[4][-1][0].strip()) is not None
                     and (line[0] + line[2]) > right - 0.05 * text_w for line in lines]
    return [
        width / height, float(np.log1p(blocks)), float(np.log1p(len(lines))), float(np.log1p(len(words))),
        text_w * text_h / (width * height), top / height, bottom / height, left / width, right / width,
        (left + right) / 2 / width, (top + bottom) / 2 / height,
        med_h / height, float(h.max() / med_h), float(np.mean(h > 1.5 * med_h)),
        float(h[first] / med_h), float(centred[first]),
        float(np.mean(centred)), float(np.mean((x1 > right - 0.03 * text_w) & (x0 > left + 0.2 * text_w))),
        float(np.mean((x0 > left + 0.03 * text_w) & (x0 < left + 0.2 * text_w))), float(np.mean(w > 0.9 * text_w)),
        float(columns(x0, text_w)), float(np.median(gaps)), float(gaps.max()),
        float(np.mean(right_numbers)), float(conf.mean()), float(np.mean(conf < 0.5)),
        float(np.mean([len(word[0]) <= 2 for word in words])) if words else 0.0,
        float(np.mean(y0 < 0.15 * height)), float(np.mean(y1 > 0.85 * height)), 1.0,
        *word_boxes(lines, med_h),
    ]


# ---------------------------------------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------------------------------------
def embed_alto(alto: Union[bytes, str]) -> np.ndarray:
    """The full vector of a page from its ALTO."""
    parsed = parse_alto(alto)
    return np.array(text_vector(alto_text(parsed)) + alto_features(parsed), dtype=np.float32)


def embed_text(text: str) -> np.ndarray:
    """The full vector of a page from its plain text (fallback without ALTO: the ALTO features are 0)."""
    return np.array(text_vector(text) + [0.0] * len(ALTO_FEATURES), dtype=np.float32)


def _embed_item(item: Tuple[str, Union[bytes, str]]) -> np.ndarray:
    kind, value = item
    return embed_alto(value) if kind == 'alto' else embed_text(value)


def _select(names: Sequence[str], blocks: Dict[str, Tuple[str, ...]], patterns: Iterable[str]) -> set:
    """Names matching block names, feature names or fnmatch patterns ('kw_*', 'head_kw:obsah')."""
    chosen = set()
    for p in patterns:
        chosen.update(blocks.get(p, ()))
        chosen.update(fnmatch.filter(names, p))
    return chosen


class PageStats:
    """The page statistics model: OCR -> full vector (FEATURES); stored full vectors -> model input (view)."""
    version = VERSION
    features = FEATURES
    grouped_features = GROUPED_FEATURES

    def __init__(self, groups: Optional[Dict[str, Sequence[str]]] = None):
        """groups: keyword group -> stems for the grouped view (default KEYWORD_GROUPS); stems must be in STEMS."""
        self.groups = {g: tuple(dict.fromkeys(s)) for g, s in (groups or KEYWORD_GROUPS).items()}
        unknown = {s for stems in self.groups.values() for s in stems} - set(STEMS)
        if unknown:
            raise ValueError(f'stems not in STEMS (new stems need re-extraction): {sorted(unknown)}')

    def embed_alto(self, alto: Union[bytes, str]) -> np.ndarray:
        return embed_alto(alto)

    def embed_text(self, text: str) -> np.ndarray:
        return embed_text(text)

    def embed(self, pages: Sequence[Union[bytes, str]], kind: str = 'alto', workers: int = 1) -> np.ndarray:
        """Full vectors of many pages (ALTO, or plain text with kind='text'), one row per page; with
        workers > 1 in parallel processes."""
        items = [(kind, p) for p in pages]
        if workers <= 1 or len(items) < 100:
            vectors = [_embed_item(i) for i in items]
        else:
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(workers) as pool:
                vectors = list(pool.map(_embed_item, items, chunksize=64))
        return np.stack(vectors) if vectors else np.zeros((0, len(FEATURES)), dtype=np.float32)

    def grouped(self, full: np.ndarray) -> np.ndarray:
        """The grouped vectors (GROUPED_FEATURES, with self.groups) of full vectors (rows)."""
        full = np.atleast_2d(np.asarray(full, dtype=np.float32))
        col = {n: i for i, n in enumerate(FEATURES)}
        parts = []
        for prefix, block in (('kw:', 'keywords'), ('head_kw:', 'head_keywords'), ('foot_kw:', 'foot_keywords')):
            hits = np.stack([full[:, [col[prefix + s] for s in stems]].sum(1) for stems in self.groups.values()], 1)
            parts.append(np.log1p(hits) if block == 'keywords' else (hits > 0).astype(np.float32))
        patterns = full[:, [col[n] for n in BLOCKS['patterns']]]
        foot_patterns = full[:, [col[n] for n in BLOCKS['foot_patterns']]]
        rest = full[:, [col[n] for n in TEXT_LINE_FEATURES + ALTO_FEATURES]]
        return np.hstack(parts + [np.log1p(patterns), (foot_patterns > 0).astype(np.float32), rest])

    def grouped_names(self) -> Tuple[str, ...]:
        groups = tuple(self.groups)
        return (tuple(f'kw_{g}' for g in groups) + tuple(f'head_{g}' for g in groups)
                + tuple(f'foot_{g}' for g in groups) + GROUPED_FEATURES[3 * len(KEYWORD_GROUPS):])

    def view(self, full: np.ndarray, grouped: bool = False, log_counts: bool = True,
             include: Optional[Iterable[str]] = None,
             exclude: Optional[Iterable[str]] = None) -> Tuple[np.ndarray, Tuple[str, ...]]:
        """(matrix, feature names) for a model from full vectors (rows, e.g. read from an LMDB).

        grouped     keyword hits summed per group (log(1 + hits) on the page, header/footer flags) and
                    patterns as log(1 + lines) and footer flags: GROUPED_FEATURES
        log_counts  without grouping, log(1 + count) for the count blocks (raw counts if False)
        include     only these blocks, feature names or fnmatch patterns ('alto_lines', 'kw_*', 'kw:obsah')
        exclude     drop these (after include)
        """
        full = np.atleast_2d(np.asarray(full, dtype=np.float32))
        if full.shape[1] != len(FEATURES):
            raise ValueError(f'{full.shape[1]} columns, page stats version {VERSION} has {len(FEATURES)}')
        if grouped:
            x, names = self.grouped(full), self.grouped_names()
            blocks = dict(GROUPED_BLOCKS, keywords=names[:len(self.groups)],
                          head_keywords=names[len(self.groups):2 * len(self.groups)],
                          foot_keywords=names[2 * len(self.groups):3 * len(self.groups)])
        else:
            x, names, blocks = full.copy(), FEATURES, BLOCKS
            if log_counts:
                counts = [i for i, n in enumerate(FEATURES) if n in COUNT_FEATURES]
                x[:, counts] = np.log1p(x[:, counts])
        keep = set(names) if include is None else _select(names, blocks, include)
        keep -= _select(names, blocks, exclude or ())
        columns_kept = [i for i, n in enumerate(names) if n in keep]
        return x[:, columns_kept], tuple(names[i] for i in columns_kept)

    def metadata(self) -> dict:
        """What to store next to the vectors (e.g. an LMDB '__metadata__' entry)."""
        return {'model': 'page_stats', 'version': VERSION, 'features': list(FEATURES),
                'keyword_groups': {g: list(s) for g, s in KEYWORD_GROUPS.items()}}


def main(argv=None):
    parser = argparse.ArgumentParser(description='Print the nonzero page statistics of one ALTO (or text) file.')
    parser.add_argument('path')
    parser.add_argument('--text', action='store_true', help='The file is plain text, not ALTO.')
    parser.add_argument('--grouped', action='store_true')
    args = parser.parse_args(argv)
    stats = PageStats()
    with open(args.path, 'rb') as f:
        data = f.read()
    full = stats.embed_text(data.decode('utf-8')) if args.text else stats.embed_alto(data)
    x, names = stats.view(full, grouped=args.grouped, log_counts=False)
    for name, value in zip(names, x[0]):
        if value:
            print(f'{name}\t{value:.4g}')


if __name__ == '__main__':
    main()
