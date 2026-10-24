"""Layout features of a page from its OCR XML (PERO OCR: PAGE XML, or ALTO v2): geometry of the page,
text area, lines and words, as a fixed vector to combine with the text cues and the embeddings.

The experiments use PAGE XML, which both storages have (ALTO exists on matylda0 only). The format
is detected from the root element. From PAGE XML a line's box is the bounding box of its Coords
polygon, its words are its text split on spaces, and every word gets the line's confidence
(TextEquiv conf); ALTO gives word boxes and word confidences (WC) directly.

All positions are relative to the page (0..1), sizes relative to the page height or to the page's
median line height, so scans of any resolution compare. Designed on PUBLIC pages only.

  page_aspect, n_blocks, n_lines, n_words          page width / height; log counts (blocks = text regions)
  text_area, text_top, text_bottom, text_left, text_right, text_center_x, text_center_y
                                                    bounding box of all lines (fractions of the page)
  line_height_med, line_height_max_ratio, large_lines   median line height / page height; tallest line /
                                                    median; share of lines over 1.5x the median (headings)
  top_line_ratio, top_line_centered                 height of the visually first line / median; whether
                                                    it is centred (title pages, headings)
  centered_lines, right_aligned_lines, indented_lines, full_width_lines   shares of the lines, relative to
                                                    the text box (title pages, poems, TOCs, indexes, prose)
  columns                                           number of text columns (clusters of line left edges)
  line_gap_med, line_gap_max                        vertical gaps between consecutive lines / median height
  right_numbers                                     share of lines ending in a number at the text box's
                                                    right edge (TOC, lists)
  conf_mean, conf_low                               mean word confidence; share of words with confidence
                                                    < 0.5 (OCR noise on images, maps, music)
  short_words                                       share of words of 1-2 characters
  top_band_lines, bottom_band_lines                 share of lines in the top / bottom 15% of the page
  has_ocr_xml                                       1 if the page has OCR XML
  word_gap_med, wide_gap_lines, char_width_ratio    word boxes: median gap between words / median line
                                                    height; share of lines with a gap over 2 line heights
                                                    (columns, TOC spacing); median character width / line
                                                    height (zeros without word boxes, e.g. PAGE XML)

WORD_CONF_FEATURES and WORD_GEOMETRY_FEATURES need words beyond the line text; LINE_FEATURES (all the
others) need only line boxes and line text, so they work with any ALTO that has lines.

Nothing here keeps text: layout_vector() maps one XML document to numbers (see CLAUDE.md for the
private pages).
"""
import re
import xml.etree.ElementTree as ET
from typing import List, Optional

import numpy as np

LAYOUT_XML_FEATURES = (
    'page_aspect', 'n_blocks', 'n_lines', 'n_words',
    'text_area', 'text_top', 'text_bottom', 'text_left', 'text_right', 'text_center_x', 'text_center_y',
    'line_height_med', 'line_height_max_ratio', 'large_lines', 'top_line_ratio', 'top_line_centered',
    'centered_lines', 'right_aligned_lines', 'indented_lines', 'full_width_lines', 'columns',
    'line_gap_med', 'line_gap_max', 'right_numbers', 'conf_mean', 'conf_low', 'short_words',
    'top_band_lines', 'bottom_band_lines', 'has_ocr_xml', 'word_gap_med', 'wide_gap_lines', 'char_width_ratio')
# the features that need words beyond the line text: word confidences and word boxes. All others use only
# the line boxes and the line text, so they work with any ALTO that has lines.
WORD_CONF_FEATURES = ('conf_mean', 'conf_low')
WORD_GEOMETRY_FEATURES = ('word_gap_med', 'wide_gap_lines', 'char_width_ratio')
LINE_FEATURES = tuple(f for f in LAYOUT_XML_FEATURES if f not in WORD_CONF_FEATURES + WORD_GEOMETRY_FEATURES)
NUMBER = re.compile(r'^[\divxlcIVXLC]+[.,]?$')
POINT = re.compile(r'(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?)')


def _local(tag: str) -> str:
    return tag.rsplit('}', 1)[-1]


def _parse_alto(root):
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


def _parse_page_xml(root):
    page = next((e for e in root.iter() if _local(e.tag) == 'Page'), None)
    if page is None:
        return None
    width, height = float(page.get('imageWidth', 0) or 0), float(page.get('imageHeight', 0) or 0)
    blocks, lines = 0, []
    for element in page.iter():
        tag = _local(element.tag)
        if tag == 'TextRegion':
            blocks += 1
        elif tag == 'TextLine':
            coords = next((c for c in element if _local(c.tag) == 'Coords'), None)
            points = POINT.findall(coords.get('points', '')) if coords is not None else []
            if not points:
                continue
            xs, ys = [float(x) for x, _ in points], [float(y) for _, y in points]
            equiv = next((t for t in element if _local(t.tag) == 'TextEquiv'), None)
            text, conf = '', 1.0
            if equiv is not None:
                conf = float(equiv.get('conf', 1) or 1)
                unicode = next((u for u in equiv if _local(u.tag) == 'Unicode'), None)
                text = (unicode.text or '') if unicode is not None else ''
            box = [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]
            if box[2] > 0 and box[3] > 0:
                lines.append((*box, [(word, conf, None, None) for word in text.split()]))
    return width, height, blocks, lines


def parse_ocr_xml(xml: bytes):
    """(page width, page height, number of blocks, lines) from PAGE XML or ALTO; a line is
    (left, top, width, height, words), a word (content, confidence, left, width); a PAGE XML word has no
    box (left and width None). None if there is no page."""
    root = ET.fromstring(xml)
    return _parse_page_xml(root) if _local(root.tag) == 'PcGts' else _parse_alto(root)


def columns(lefts: np.ndarray, text_width: float) -> int:
    """Number of clusters of line left edges at least a quarter of the text width apart."""
    if len(lefts) < 3:
        return 1 if len(lefts) else 0
    edges = np.sort(lefts)
    gaps = np.diff(edges) > 0.25 * text_width
    starts = np.split(edges, np.flatnonzero(gaps) + 1)
    return int(sum(len(s) >= max(3, 0.1 * len(edges)) for s in starts)) or 1


def layout_vector(xml: Optional[bytes]) -> List[float]:
    zeros = [0.0] * len(LAYOUT_XML_FEATURES)
    if xml is None:
        return zeros
    parsed = parse_ocr_xml(xml)
    if parsed is None or parsed[0] <= 0 or parsed[1] <= 0:
        return zeros
    width, height, blocks, lines = parsed
    if not lines:
        v = zeros.copy()
        v[0], v[1], v[-1] = width / height, float(np.log1p(blocks)), 1.0
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
        *word_geometry(lines, med_h),
    ]


def word_geometry(lines, med_h: float) -> List[float]:
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


def ocr_text_lines(xml: bytes) -> List[str]:
    """The line texts, top to bottom by position on the page (for the text cues)."""
    parsed = parse_ocr_xml(xml)
    if parsed is None:
        return []
    return [' '.join(word[0] for word in line[4]) for line in sorted(parsed[3], key=lambda line: line[1])]


EDGE_METHODS = ('first3', 'gap1', 'gap', 'band', 'gap+size')


def header_footer_counts(xml: bytes, method: str = 'gap') -> tuple:
    """(header lines, footer lines) of a page, counted on its lines sorted top to bottom (as ocr_text_lines).

      first3     the first and last 3 lines (as for plain text)
      gap1       the lines above the first vertical gap of over 1.5 median line heights from the top,
                 within the top 6 lines, and below the first such gap from the bottom, within the bottom 6;
                 without such a gap, the lines in the top (bottom) 10% of the text box
      gap        as gap1 but the last such gap within the top (bottom) 6 lines, so that a running header
                 or page number and a heading below it are both in the header
      band       the lines starting in the top 15% of the text box, and ending in its bottom 15%
      gap+size   gap, extended by the consecutive tall lines (over 1.3x the median) that follow it
    """
    parsed = parse_ocr_xml(xml)
    if parsed is None or not parsed[3]:
        return 0, 0
    lines = sorted(parsed[3], key=lambda line: line[1])
    n = len(lines)
    if method == 'first3':
        return min(3, n), (3 if n > 3 else 0)
    top = np.array([line[1] for line in lines])
    bottom = top + np.array([line[3] for line in lines])
    height = np.array([line[3] for line in lines])
    med_h = float(np.median(height))
    text_top, text_bottom = top.min(), bottom.max()
    text_h = max(text_bottom - text_top, 1.0)
    if method == 'band':
        head = int(np.sum(top < text_top + 0.15 * text_h))
        foot = int(np.sum(bottom > text_bottom - 0.15 * text_h))
        return head, min(foot, n - head)
    gaps = top[1:] - bottom[:-1]  # gaps[i]: between line i and line i + 1
    big_top = [i for i in range(min(6, n - 1)) if gaps[i] > 1.5 * med_h]
    big_bottom = [i for i in range(n - 2, max(n - 8, -1), -1) if gaps[i] > 1.5 * med_h]  # from the bottom up
    if big_top:
        head = (big_top[0] if method == 'gap1' else big_top[-1]) + 1
    else:
        head = max(1, int(np.sum(top < text_top + 0.10 * text_h)))
    if big_bottom:
        foot = n - 1 - (big_bottom[0] if method == 'gap1' else big_bottom[-1])
    else:
        foot = int(np.sum(bottom > text_bottom - 0.10 * text_h))
    if method == 'gap+size':
        while head < n and height[head] > 1.3 * med_h:
            head += 1
    return head, max(0, min(foot, n - head))
