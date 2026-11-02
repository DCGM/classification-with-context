import json

import pytest

from tools.start_store_experiment import store_ready


def test_store_gate_requires_publication_and_completed_validation(tmp_path):
    failure = tmp_path / 'failure.json'
    assert not store_ready(tmp_path, [failure])
    (tmp_path / 'manifest.json').write_text(json.dumps({'complete': True}))
    assert not store_ready(tmp_path, [failure])
    status = tmp_path / 'status.json'
    status.write_text(json.dumps({'phase': 'validating'}))
    assert not store_ready(tmp_path, [failure])
    status.write_text(json.dumps({'phase': 'complete'}))
    assert store_ready(tmp_path, [failure])
    failure.write_text('{}')
    with pytest.raises(RuntimeError, match='Data preparation failed'):
        store_ready(tmp_path, [failure])
