from pathlib import Path

import pytest

from training_store.relocate import relocate


def test_relocation_verifies_existing_files_and_preserves_path(tmp_path):
    source, dest = tmp_path / 'source', tmp_path / 'archive'
    source.mkdir()
    dest.mkdir()
    (source / 'data').write_bytes(b'embeddings' * 100)
    (dest / 'data').write_bytes((source / 'data').read_bytes())
    journal = tmp_path / 'journal.json'
    relocate(source, dest, journal)
    assert source.is_symlink()
    assert (source / 'data').read_bytes() == b'embeddings' * 100
    relocate(source, dest, journal)
    assert not list(tmp_path.glob('*.verified-relocation'))


def test_different_existing_archive_never_removes_source(tmp_path):
    source, dest = tmp_path / 'source', tmp_path / 'archive'
    source.mkdir()
    dest.mkdir()
    (source / 'data').write_bytes(b'new')
    (dest / 'data').write_bytes(b'old')
    with pytest.raises(ValueError, match='differs'):
        relocate(source, dest, tmp_path / 'journal.json')
    assert not source.is_symlink()
    assert (source / 'data').read_bytes() == b'new'


def test_verified_copy_keeps_both_physical_copies(tmp_path):
    source, dest = tmp_path / 'nvme', tmp_path / 'ssd.copying'
    source.mkdir()
    (source / 'records.bin').write_bytes(b'raw embeddings')
    journal = tmp_path / 'copy.json'
    relocate(source, dest, journal, replace_source=False)
    assert source.is_dir() and not source.is_symlink()
    assert (dest / 'records.bin').read_bytes() == (source / 'records.bin').read_bytes()
    relocate(source, dest, journal, replace_source=False)
