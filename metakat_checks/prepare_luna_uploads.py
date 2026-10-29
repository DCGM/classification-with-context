"""Build missing Luna upload ZIPs and audit all three resource-only bundles.

Run with MetaKat's Python and biblio-spans on PYTHONPATH. No API calls or model
inference. Configurations and source bundle directories are left unchanged.
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import tempfile
import zipfile

import yaml

ROOT = Path('/mnt/kolosus/data/metakat_worker/src_engines')


def digest(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def resources(config, base):
    selected = set()

    def visit(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key.endswith(('_path', '_dir', '_paths', '_dirs')) and child is not None:
                    for item in child if isinstance(child, list) else [child]:
                        relative = Path(item)
                        if relative.is_absolute() or '..' in relative.parts:
                            raise ValueError(f'Nonportable resource path: {relative}')
                        path = base / relative
                        if not path.exists():
                            raise FileNotFoundError(path)
                        files = path.rglob('*') if path.is_dir() else [path]
                        for file in files:
                            if file.is_symlink():
                                raise ValueError(f'Symlink in bundle: {file}')
                            if file.is_file():
                                if any(p in ('.cache', '__pycache__', '.git') for p in file.relative_to(base).parts):
                                    raise ValueError(f'Cache/source metadata in resource: {file}')
                                selected.add(file.relative_to(base).as_posix())
                else:
                    visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(config)
    return selected


def main():
    import metakat
    import text_geometry_aligner
    from metakat.engine_config import prepare_engine_config
    from metakat.process_batch import _preflight_engine_requirements
    from metakat.biblio.engines.core.definitions import load_biblio_core_engine
    from metakat.chapter.engines.core.chapter_extraction.engine_vlm import ChapterExtractionEngineVLM

    source = Path(metakat.__file__).resolve().parents[1]
    aligner = Path(text_geometry_aligner.__file__).resolve().parents[1]
    revision = lambda path: subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
    reports = []
    for kind in ('monograph', 'periodical', 'chapter'):
        name = f'{kind}.openai.gpt-6-luna.medium'
        base = ROOT / f'{name}.v1.7.0'
        print(f'Checking {base.name}', flush=True)
        config = json.loads((base / 'engine_config.json').read_text())
        assert yaml.safe_load((base / 'engine_config.yaml').read_text()) == config
        descriptor_path = ROOT / f'{base.name}.engine.json'
        descriptor = json.loads(descriptor_path.read_text())
        assert descriptor['name'] == name and descriptor['version'] == 'v1.7.0'
        assert descriptor['definition'] == config
        expected = resources(config, base) | {'engine_config.json', 'engine_config.yaml'}
        actual = {p.relative_to(base).as_posix() for p in base.rglob('*') if p.is_file()}
        if actual != expected:
            raise ValueError(f'Unexpected files: {actual - expected}; missing: {expected - actual}')
        # Every checkpoint outside the explicitly referenced HF directory must be
        # named directly by config. The current bundles need only these YOLOs.
        yolos = sorted(p for p in expected if p.endswith('/best.pt'))
        assert len(yolos) == (2 if kind == 'chapter' else 1), yolos
        assert any(p.startswith('page_number/') for p in yolos)
        if kind == 'chapter':
            assert any('/chapter_page_analysis_engine_yolo_alto.' in p for p in yolos)
        assert not any(p.startswith('biblio/') for p in expected)
        assert not any('chapter_extraction_engine_yolo' in p for p in expected)
        resolved = prepare_engine_config(config, base_dir=base, require_within_base=True)
        _preflight_engine_requirements(resolved)
        bibliography = load_biblio_core_engine(resolved['biblio']['core'])
        assert bibliography.process(()).spans == ()
        if kind == 'chapter':
            toc = ChapterExtractionEngineVLM(resolved['chapter']['core']['extraction'], client=object())
            assert toc.process(()).chapters == ()
        metadata = {}
        for relative in sorted(expected):
            path = base / relative
            metadata[relative] = {'bytes': path.stat().st_size, 'sha256': digest(path)}
        assert metadata[config['page_type']['core']['image_checkpoint_path']]['sha256'] == config['page_type']['core']['image']['expected_sha256']
        archive = ROOT / f'{base.name}.zip'
        rebuild = not archive.exists()
        if not rebuild:
            with zipfile.ZipFile(archive) as bundle:
                rebuild = set(bundle.namelist()) != expected or any(
                    info.file_size != metadata[info.filename]['bytes'] for info in bundle.infolist()
                ) or json.loads(bundle.read('engine_config.json')) != config
        if rebuild:
            print(f'Building {archive.name}: {len(expected)} files', flush=True)
            with tempfile.NamedTemporaryFile(prefix='.' + archive.name, suffix='.tmp', dir=ROOT, delete=False) as temporary:
                staged = Path(temporary.name)
            try:
                with zipfile.ZipFile(staged, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as bundle:
                    for relative in sorted(expected):
                        bundle.write(base / relative, relative)
                with zipfile.ZipFile(staged) as bundle:
                    assert bundle.testzip() is None
                staged.chmod(0o644)
                staged.replace(archive)
            finally:
                staged.unlink(missing_ok=True)
        print(f'Verifying all archived bytes: {archive.name}', flush=True)
        with zipfile.ZipFile(archive) as bundle:
            assert len(bundle.infolist()) == len(expected)
            for relative in sorted(expected):
                with bundle.open(relative) as content:
                    assert hashlib.file_digest(content, 'sha256').hexdigest() == metadata[relative]['sha256'], relative
        report = dict(name=name, version='v1.7.0', status='ready', archive=str(archive),
                      checked_utc=datetime.now(timezone.utc).isoformat(),
                      archive_bytes=archive.stat().st_size, archive_sha256=digest(archive),
                      descriptor=str(descriptor_path), descriptor_sha256=digest(descriptor_path), files=metadata,
                      yolo_weights=yolos, metakat_commit=revision(source), aligner_commit=revision(aligner),
                      validation=['JSON/YAML/descriptor equality', 'all and only configured resources',
                                  'portable resource paths', 'pipeline dependency preflight',
                                  'VLM constructors and empty-input paths; no requests',
                                  'DINO checksum', 'SHA256 of every archived file matches bundle'],
                      runtime_requirements=['Updated MetaKat and text-geometry-aligner revisions recorded here',
                                            'OPENROUTER_API_KEY supplied by the worker environment',
                                            'CUDA runtime for configured local models'],
                      published=False, paid_requests=0)
        (ROOT / f'{base.name}.upload-check.json').write_text(json.dumps(report, indent=2) + '\n')
        reports.append(report)
        print(f'Ready: {archive.name}, {archive.stat().st_size:,} bytes; YOLOs={len(yolos)}', flush=True)
    (ROOT / 'luna-v1.7.0-upload-check.json').write_text(json.dumps(reports, indent=2) + '\n')


if __name__ == '__main__':
    main()
