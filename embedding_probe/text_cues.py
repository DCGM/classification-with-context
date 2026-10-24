"""Cheap text cues of a page: keyword groups, regex patterns and layout statistics of its OCR text,
as a fixed vector to combine with the page embeddings.

Designed on PUBLIC text only (explore_public_text.py, on the annotated trn pages and the DB-labelled
public sample of build_dev_sample.py). The keywords are lower case without diacritics and match the
start of a word (OCR and inflection variants: 'predmluv' matches 'předmluva', 'předmluvou').

The vector (CUE_FEATURES names its entries):
  kw_<group>       log(1 + number of keyword hits) on the whole page
  head_<group>     1 if a keyword of the group is in the first 3 non-empty lines (headings)
  foot_<group>     1 if a keyword of the group is in the last 3 non-empty lines (footers: imprints, end marks)
  pat_<pattern>    log(1 + number of matching lines) of a regex pattern (ISBN, price, year, ...)
  foot_pat_<p>     1 if the pattern matches one of the last 3 lines
  layout stats     the 17 line statistics of text_baselines.py, plus alphabetical line order
                   (indexes), numeric lines and multi-number lines (tables, TOCs)

Nothing here reads text: cue_vector() maps one text to numbers, so it can run on private text in
memory, inside a script that writes only predictions (see CLAUDE.md).
"""
import re
import unicodedata
from typing import Dict, List, Sequence, Tuple

import numpy as np

from embedding_probe.text_baselines import LAYOUT_FEATURES, layout_features

# group -> keyword prefixes (lower case, no diacritics); several words are a phrase matched as such
KEYWORDS: Dict[str, Tuple[str, ...]] = {
    'toc': ('obsah', 'inhalt', 'contents', 'table des matieres', 'sommaire', 'spis tresci', 'tartalom',
            'содержание', 'оглавление', 'index capitum', 'indice'),
    'index': ('rejstrik', 'registrik', 'rejstr', 'register', 'index', 'sachregister', 'namenregister',
              'jmenny', 'vecny', 'ukazatel', 'указатель', 'viz '),
    'errata': ('errata', 'corrigenda', 'oprav', 'tiskov', 'druckfehler', 'berichtigung', 'rettelser', 'omyl',
               'radek', 'shora', 'zdola', 'misto ', 'cti ', 'lies ', 'statt ', 'zeile', 'additions'),
    'preface': ('predmluv', 'predml', 'vorrede', 'vorwort', 'preface', 'praefatio', 'prefatio', 'praefatio',
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

EXTRA_LAYOUT = ('alphabetical_lines', 'numeric_lines', 'multi_number_lines', 'first_line_caps')
EDGE_LINES = 3  # heading = first 3, footer = last 3 non-empty lines (chosen on public text, see the README)
CUE_FEATURES = (tuple(f'kw_{g}' for g in KEYWORDS) + tuple(f'head_{g}' for g in KEYWORDS)
                + tuple(f'foot_{g}' for g in KEYWORDS) + tuple(f'pat_{p}' for p in PATTERNS)
                + tuple(f'foot_pat_{p}' for p in PATTERNS) + LAYOUT_FEATURES + EXTRA_LAYOUT)

_WORD_START = {g: [re.compile(r'(?<![^\W\d_])' + re.escape(k)) for k in keys] for g, keys in KEYWORDS.items()}


def fold(text: str) -> str:
    """Lower case without diacritics, punctuation kept."""
    text = unicodedata.normalize('NFKD', text.lower().replace('ß', 'ss'))
    return ''.join(c for c in text if not unicodedata.combining(c))


def normalize(text: str) -> str:
    """fold() with punctuation inside lines turned into spaces (so 'Č. inv.:' -> 'c inv '), for keywords."""
    return re.sub(r'[^\w\n]+', ' ', fold(text))


def _hits(text: str, group: str) -> int:
    return sum(len(p.findall(text)) for p in _WORD_START[group])


def extra_layout(lines: Sequence[str]) -> List[float]:
    if not lines:
        return [0.0] * len(EXTRA_LAYOUT)
    firsts = [line.lstrip()[0].lower() for line in lines if line.strip() and line.lstrip()[0].isalpha()]
    ordered = sum(a <= b for a, b in zip(firsts, firsts[1:])) / max(len(firsts) - 1, 1) if len(firsts) > 2 else 0.0
    digits = [sum(c.isdigit() for c in line) / max(len(line.replace(' ', '')), 1) for line in lines]
    numbers = [len(re.findall(r'\d+', line)) for line in lines]
    first = lines[0]
    return [ordered, float(np.mean([d >= 0.5 for d in digits])), float(np.mean([n >= 3 for n in numbers])),
            float(sum(c.isalpha() for c in first) >= 3 and first.upper() == first)]


def cue_vector(text: str, head_lines: int = None, foot_lines: int = None) -> List[float]:
    """The cue vector of a page text (one line per text line, top to bottom). The heading is the first
    head_lines non-empty lines and the footer the last foot_lines (default EDGE_LINES each, a page of up to
    EDGE_LINES lines having no separate footer); with OCR geometry they can come from the layout instead
    (ocr_layout.header_footer_counts)."""
    raw_lines = [line.strip() for line in text.split('\n') if line.strip()]
    folded_lines = [line.strip() for line in fold(text).split('\n') if line.strip()]
    norm_lines = [line.strip() for line in normalize(text).split('\n') if line.strip()]
    if head_lines is None:
        head_lines = EDGE_LINES
    if foot_lines is None:
        foot_lines = EDGE_LINES if len(norm_lines) > EDGE_LINES else 0
    padded = ' ' + ' '.join(norm_lines) + ' '  # trailing space lets 'viz ' match at a line end
    head = ' ' + ' '.join(norm_lines[:head_lines]) + ' '
    foot = ' ' + ' '.join(norm_lines[len(norm_lines) - foot_lines:]) + ' ' if foot_lines else ' '
    foot_folded = folded_lines[len(folded_lines) - foot_lines:] if foot_lines else []
    kw = [float(np.log1p(_hits(padded, g))) for g in KEYWORDS]
    head_kw = [float(_hits(head, g) > 0) for g in KEYWORDS]
    foot_kw = [float(_hits(foot, g) > 0) for g in KEYWORDS]
    pat = [float(np.log1p(sum(1 for line in folded_lines if p.search(line)))) for p in PATTERNS.values()]
    foot_pat = [float(any(p.search(line) for line in foot_folded)) for p in PATTERNS.values()]
    return kw + head_kw + foot_kw + pat + foot_pat + layout_features(text) + extra_layout(raw_lines)


def _cue_vector_item(item) -> List[float]:
    return cue_vector(item) if isinstance(item, str) else cue_vector(*item)


def cue_matrix(texts: Sequence, workers: int = None) -> np.ndarray:
    """Cue vectors of many texts (or (text, head_lines, foot_lines) tuples), computed by parallel worker
    processes (the texts are passed in memory)."""
    import os
    from concurrent.futures import ProcessPoolExecutor
    workers = workers or min(os.cpu_count() or 1, 12)
    if workers <= 1 or len(texts) < 2000:
        return np.array([_cue_vector_item(t) for t in texts], dtype=np.float32)
    with ProcessPoolExecutor(workers) as pool:
        return np.array(list(pool.map(_cue_vector_item, texts, chunksize=500)), dtype=np.float32)
