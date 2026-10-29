from metakat_checks.recto_verso import (SETTING, engine_settings, is_sheet_label, recto_verso_documents,
                                        recto_verso_field)


def pages(document, labels):
    return [{'document': document, 'page_number': label} for label in labels]


def test_sheet_labels():
    assert all(map(is_sheet_label, ['12r', '12v', '[12v]', 'IIr', '[iv]', 'ir']))
    assert not any(map(is_sheet_label, ['12', '[12]', '[12a]', 'A 1r', 'F1r', '55 [58]', '', None, 'v']))


def test_a_document_needs_enough_sheet_labels_and_a_share_of_the_numbered_pages():
    sheets = [f'{n}{side}' for n in range(1, 5) for side in 'rv']
    assert recto_verso_documents(pages('a', sheets)) == {'a': True}
    assert recto_verso_documents(pages('a', sheets[:4])) == {'a': False}  # too few
    assert recto_verso_documents(pages('a', sheets + [str(n) for n in range(1, 40)])) == {'a': False}  # a minority
    assert recto_verso_documents(pages('a', sheets + [''] * 30)) == {'a': True}  # unnumbered pages do not count


def test_a_batch_is_in_the_mode_when_any_document_is_and_the_settings_follow():
    sheets = [f'{n}{side}' for n in range(1, 5) for side in 'rv']
    mixed = pages('a', sheets) + pages('b', [str(n) for n in range(1, 20)])
    field = recto_verso_field(mixed)
    assert field['batch'] and field['documents'] == {'a': True, 'b': False}
    assert engine_settings({'recto_verso': field}) == [(SETTING, 'true')]
    assert engine_settings({'recto_verso': recto_verso_field(pages('b', ['1', '2']))}) == []
    assert engine_settings({}) == []


def test_base_number_reads_the_number_in_any_convention():
    from metakat_checks.page_number_report import base_number
    assert {base_number(x) for x in ['12', '[12]', '12r', '[12v]', '[12a]']} == {'12'}
    assert base_number('iv') == base_number('[IV]') == 'iv'
    assert [base_number(x) for x in ['FC', 'G9', 'A 1r', '5,6', '55 [58]', '[a]', '', None]] == [None] * 8
