from embeddings.text.normalization import normalize_ocr_lines, normalize_ocr_words


def test_blank_and_whitespace_only_lines_are_dropped():
    assert normalize_ocr_lines('Obsah\n\n   \n\t\nKapitola 1\n') == 'Obsah\nKapitola 1'


def test_words_are_separated_by_one_space():
    assert normalize_ocr_lines('  a \t b\u00a0c  \nd') == 'a b c\nd'


def test_page_without_text_is_empty_whatever_whitespace_it_has():
    assert normalize_ocr_lines('') == normalize_ocr_lines('\n \n\n') == normalize_ocr_lines(' \u00a0\t') == ''


def test_cap_counts_characters_of_the_normalized_text():
    text = '\n'.join(['\n\nabcd'] * 3)
    assert normalize_ocr_lines(text, max_chars=7) == 'abcd\nab'
    assert normalize_ocr_lines(text, max_chars=None) == 'abcd\nabcd\nabcd'


def test_words_only_text_has_no_line_breaks_and_one_space_between_words():
    assert normalize_ocr_words('Obsah\n\n   \n\t\nKapitola  1\n') == 'Obsah Kapitola 1'
    assert normalize_ocr_words('\n \n\n') == ''
    assert normalize_ocr_words('a b\ncd', max_chars=4) == 'a b '
