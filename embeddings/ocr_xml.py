"""Simple readers of OCR page XML (ALTO, PAGE XML) into one page model, and a PAGE XML -> ALTO converter.
Standard library only.

The model keeps what the page statistics and similar tools need: the page size, the text blocks, and per
line its box, confidence, text and words with their horizontal positions.

  read_alto(xml)       ALTO (any version; namespaces are ignored)
  read_page_xml(xml)   PAGE XML: a TextRegion is a block; a line's box is the bounding box of its Coords
                       polygon, its text the TextEquiv/Unicode and its confidence the TextEquiv conf. PAGE
                       XML has no word boxes: the words are placed along the line by their length
                       (place_words) and get the line's confidence. As in PERO OCR's ALTO export, lines
                       without a Baseline or without text are left out and the box is cut to whole pixels,
                       so the page statistics equal those of PERO's ALTO with words placed by length.
  read_ocr(xml)        either, by the root element
  to_alto(page)        ALTO v2 of a page model, e.g. to store PAGE XML pages as ALTO; reading it back with
                       read_alto() gives the same model

    page = read_ocr(xml_bytes)
    alto_bytes = to_alto(read_page_xml(page_xml_bytes))
"""
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import List, Optional, Union

ALTO_NS = 'http://www.loc.gov/standards/alto/ns-v2#'
POINT = re.compile(r'(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?)')


@dataclass
class Word:
    text: str
    conf: float = 1.0
    left: Optional[float] = None    # None: no word box
    width: Optional[float] = None


@dataclass
class Line:
    left: float
    top: float
    width: float
    height: float
    words: List[Word] = field(default_factory=list)
    conf: Optional[float] = None

    @property
    def text(self) -> str:
        return ' '.join(w.text for w in self.words)


@dataclass
class Block:
    lines: List[Line] = field(default_factory=list)


@dataclass
class OcrPage:
    width: float
    height: float
    blocks: List[Block] = field(default_factory=list)

    @property
    def lines(self) -> List[Line]:
        return [line for block in self.blocks for line in block.lines]


def _local(tag: str) -> str:
    return tag.rsplit('}', 1)[-1]


def _children(element, name: str):
    return [c for c in element if _local(c.tag) == name]


def _float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def read_alto(xml: Union[bytes, str]) -> Optional[OcrPage]:
    """The page model of ALTO; None without a Page element. Lines without a positive box are left out."""
    root = ET.fromstring(xml)
    page_el = next((e for e in root.iter() if _local(e.tag) == 'Page'), None)
    if page_el is None:
        return None
    page = OcrPage(_float(page_el.get('WIDTH')), _float(page_el.get('HEIGHT')))
    for block_el in (e for e in page_el.iter() if _local(e.tag) == 'TextBlock'):
        block = Block()
        for line_el in _children(block_el, 'TextLine'):
            box = [_float(line_el.get(k)) for k in ('HPOS', 'VPOS', 'WIDTH', 'HEIGHT')]
            if box[2] <= 0 or box[3] <= 0:
                continue
            words = [Word(s.get('CONTENT', ''), _float(s.get('WC'), 1.0) if s.get('WC') else 1.0,
                          _float(s.get('HPOS')), _float(s.get('WIDTH')))
                     for s in _children(line_el, 'String')]
            block.lines.append(Line(*box, words=words))
        page.blocks.append(block)
    return page


def place_words(line: Line) -> None:
    """Places the line's words along it by their length in characters (one character of space between
    words), in whole pixels, and gives them the line's confidence."""
    if not line.words:
        return
    lengths = [max(len(w.text), 1) for w in line.words]
    unit = line.width / (sum(lengths) + len(lengths) - 1)
    x = line.left
    for word, n in zip(line.words, lengths):
        word.left, word.width = float(round(x)), float(round(n * unit))
        word.conf = line.conf if line.conf is not None else 1.0
        x += n * unit
        x += unit


def read_page_xml(xml: Union[bytes, str]) -> Optional[OcrPage]:
    """The page model of PAGE XML (words placed by length); None without a Page element."""
    root = ET.fromstring(xml)
    page_el = next((e for e in root.iter() if _local(e.tag) == 'Page'), None)
    if page_el is None:
        return None
    page = OcrPage(_float(page_el.get('imageWidth')), _float(page_el.get('imageHeight')))
    for region in (e for e in page_el.iter() if _local(e.tag) == 'TextRegion'):
        block = Block()
        for line_el in _children(region, 'TextLine'):
            coords = next(iter(_children(line_el, 'Coords')), None)
            points = POINT.findall(coords.get('points', '')) if coords is not None else []
            if not points or not _children(line_el, 'Baseline'):
                continue
            xs, ys = [float(x) for x, _ in points], [float(y) for _, y in points]
            box = [float(int(min(xs))), float(int(min(ys))), float(int(max(xs) - min(xs))),
                   float(int(max(ys) - min(ys)))]
            equiv = next(iter(_children(line_el, 'TextEquiv')), None)
            text, conf = '', None
            if equiv is not None:
                if equiv.get('conf') is not None:
                    conf = _float(equiv.get('conf'), 1.0)
                unicode_el = next(iter(_children(equiv, 'Unicode')), None)
                text = (unicode_el.text or '') if unicode_el is not None else ''
            if not text.strip() or box[2] <= 0 or box[3] <= 0:
                continue
            line = Line(*box, words=[Word(t) for t in text.split()], conf=conf)
            place_words(line)
            block.lines.append(line)
        page.blocks.append(block)
    return page


def read_ocr(xml: Union[bytes, str]) -> Optional[OcrPage]:
    """The page model of ALTO or PAGE XML, by the root element."""
    head = xml[:2000] if isinstance(xml, bytes) else xml[:2000].encode('utf-8')
    return read_page_xml(xml) if b'PcGts' in head else read_alto(xml)


def _number(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f'{value:.2f}'


def to_alto(page: OcrPage, page_id: str = 'page') -> bytes:
    """ALTO v2 of a page model (TextBlock per block, TextLine boxes, String per word with its box and WC)."""
    ET.register_namespace('', ALTO_NS)

    def el(parent, name, **attrs):
        return ET.SubElement(parent, f'{{{ALTO_NS}}}{name}', {k: str(v) for k, v in attrs.items()})

    root = ET.Element(f'{{{ALTO_NS}}}alto')
    layout = el(root, 'Layout')
    page_el = el(layout, 'Page', ID=page_id, WIDTH=_number(page.width), HEIGHT=_number(page.height),
                 PHYSICAL_IMG_NR='1')
    space = el(page_el, 'PrintSpace', HPOS='0', VPOS='0', WIDTH=_number(page.width), HEIGHT=_number(page.height))
    for b, block in enumerate(page.blocks):
        block_el = el(space, 'TextBlock', ID=f'block_{b}')
        for n, line in enumerate(block.lines):
            line_el = el(block_el, 'TextLine', ID=f'line_{b}_{n}', HPOS=_number(line.left), VPOS=_number(line.top),
                         WIDTH=_number(line.width), HEIGHT=_number(line.height))
            for w, word in enumerate(line.words):
                attrs = {'CONTENT': word.text, 'WC': repr(float(word.conf))}
                if word.left is not None and word.width is not None:
                    attrs.update(HPOS=_number(word.left), VPOS=_number(line.top), WIDTH=_number(word.width),
                                 HEIGHT=_number(line.height))
                if w:
                    el(line_el, 'SP')
                el(line_el, 'String', **attrs)
    return ET.tostring(root, encoding='utf-8', xml_declaration=True)
