import zipfile

import numpy as np
import pytest

from embeddings.ocr_xml import read_page_xml, to_alto
from embeddings.tests.test_ocr_xml import PAGE_XML
from embeddings.text.page_stats.extract_to_lmdb import StorageUnavailable, WorkUnit, process_zip
from embeddings.text.page_stats.page_stats import embed_ocr, from_sparse


def write_zip(path, members):
    with zipfile.ZipFile(path, 'w') as z:
        for name, data in members.items():
            z.writestr(name, data)


def test_pages_missing_in_the_alto_zip_come_from_the_page_xml_zip(tmp_path):
    alto = to_alto(read_page_xml(PAGE_XML), 'a')
    write_zip(tmp_path / 'doc.alto_xml.zip', {'a.xml': alto})
    write_zip(tmp_path / 'doc.page_xml.zip', {'a.xml': PAGE_XML, 'b.xml': PAGE_XML})
    pages = [('lib', 'a'), ('lib', 'b'), ('lib', 'c')]
    results = process_zip(WorkUnit(str(tmp_path / 'doc.page_xml.zip'), pages))
    assert [(r.page_id, r.status, r.reason) for r in results] == [
        ('a', 'done_alto', ''), ('b', 'done_page_xml', ''), ('c', 'missing', 'member_missing')]
    np.testing.assert_array_equal(from_sparse(results[1].value), from_sparse(results[0].value))
    np.testing.assert_allclose(from_sparse(results[1].value), embed_ocr(PAGE_XML).astype(np.float16))


def test_page_xml_zip_only_and_no_zips(tmp_path):
    write_zip(tmp_path / 'doc.page_xml.zip', {'a.xml': PAGE_XML})
    results = process_zip(WorkUnit(str(tmp_path / 'doc.page_xml.zip'), [('lib', 'a')]))
    assert results[0].status == 'done_page_xml'
    results = process_zip(WorkUnit(str(tmp_path / 'other.page_xml.zip'), [('lib', 'a')]))
    assert (results[0].status, results[0].reason) == ('missing', 'archive_missing')


def test_a_vanished_storage_stops_instead_of_recording_missing_pages(tmp_path):
    with pytest.raises(StorageUnavailable):
        process_zip(WorkUnit(str(tmp_path / 'unmounted' / 'doc.page_xml.zip'), [('lib', 'a')]))
