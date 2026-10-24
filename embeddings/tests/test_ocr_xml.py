import numpy as np
import pytest

from embeddings.ocr_xml import read_alto, read_ocr, read_page_xml, to_alto

PAGE_XML = '''<?xml version="1.0" encoding="UTF-8"?>
<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15">
<Page imageFilename="p1.jpg" imageWidth="1000" imageHeight="2000">
<TextRegion id="r1"><Coords points="300,100 700,100 700,180 300,180"/>
<TextLine id="l1"><Coords points="300,100 700,100 700,180 300,180"/><Baseline points="300,170 700,170"/>
<TextEquiv conf="0.912345"><Unicode>OBSAH</Unicode></TextEquiv></TextLine></TextRegion>
<TextRegion id="r2"><Coords points="100,300 900,300 900,400 100,400"/>
<TextLine id="l2"><Coords points="100,300 900,301.6 899,340.7 100,340"/><Baseline points="100,335 900,335"/>
<TextEquiv conf="1.0"><Unicode>Kapitola první 5</Unicode></TextEquiv></TextLine>
<TextLine id="l3"><Coords points="100,360 900,360 900,400 100,400"/><Baseline points="100,395 900,395"/>
<TextEquiv conf="0.2"><Unicode>Rejstřík 12</Unicode></TextEquiv></TextLine>
<TextLine id="no_text"><Coords points="100,500 900,500 900,540 100,540"/><Baseline points="100,535 900,535"/>
<TextEquiv conf="0.9"><Unicode> </Unicode></TextEquiv></TextLine>
<TextLine id="no_baseline"><Coords points="100,600 900,600 900,640 100,640"/>
<TextEquiv conf="0.9"><Unicode>text</Unicode></TextEquiv></TextLine></TextRegion>
</Page></PcGts>'''.encode('utf-8')


def test_read_page_xml_places_words_by_length():
    page = read_page_xml(PAGE_XML)
    assert (page.width, page.height, len(page.blocks), len(page.lines)) == (1000, 2000, 2, 3)  # as PERO: no text, no baseline -> out
    line = page.lines[1]
    assert (line.left, line.top, line.width, line.height) == (100, 300, 800, 40) and line.text == 'Kapitola první 5'
    words = line.words
    assert [w.text for w in words] == ['Kapitola', 'první', '5'] and all(w.conf == 1.0 for w in words)
    assert words[0].left == 100 and words[-1].left + words[-1].width == pytest.approx(900, abs=1)
    assert words[0].width > words[1].width > words[2].width  # by length
    assert page.lines[0].words[0].conf == 0.912345


def test_alto_round_trip_and_format_detection():
    page = read_page_xml(PAGE_XML)
    alto = to_alto(page)
    assert read_ocr(alto) == read_alto(alto) and read_ocr(PAGE_XML) == page
    again = read_alto(alto)
    assert [(l.left, l.top, l.width, l.height, l.text) for l in again.lines] == \
        [(l.left, l.top, l.width, l.height, l.text) for l in page.lines]
    assert [(w.text, w.conf, w.left, w.width) for l in again.lines for w in l.words] == \
        [(w.text, w.conf, w.left, w.width) for l in page.lines for w in l.words]
    assert to_alto(again) == alto


def test_page_stats_of_page_xml_equal_those_of_its_alto():
    from embeddings.text.page_stats.page_stats import embed_ocr
    np.testing.assert_array_equal(embed_ocr(PAGE_XML), embed_ocr(to_alto(read_page_xml(PAGE_XML))))
    assert embed_ocr(PAGE_XML).any()
