"""Package the current chapter pipeline with Luna title-page and TOC reading.

Run with MetaKat's Python and its worker checkout on PYTHONPATH.
Only bundle files are written; no API, worker or model inference is invoked.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import zipfile

import yaml

NAME='chapter.openai.gpt-6-luna.medium'
VERSION='v1.7.0'


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('/mnt/kolosus/data/metakat_worker/src_engines'))
    args=parser.parse_args()
    root=args.root.resolve()
    destination=root/f'{NAME}.{VERSION}'
    descriptor_path=root/f'{destination.name}.engine.json'
    preparation_path=root/f'{destination.name}.preparation.json'
    archive_path=root/f'{destination.name}.zip'
    if any(p.exists() for p in (destination,descriptor_path,preparation_path,archive_path)):
        raise FileExistsError('Refusing to overwrite an existing bundle')
    monograph=root/f'monograph.openai.gpt-6-luna.medium.{VERSION}'
    chapter=root/f'chapter.{VERSION}'
    config=json.loads((monograph/'engine_config.json').read_text())
    old_chapter=json.loads((chapter/'engine_config.json').read_text())['chapter']
    config['biblio']['core']['vlm']['max_attempts']=1
    config['chapter']=copy.deepcopy(old_chapter)
    config['chapter']['core']['extraction']={
        'name':'chapter_extraction_engine_vlm',
        'vlm':copy.deepcopy(config['biblio']['core']['vlm']),
        'confidence':.95,
    }
    descriptor={
        'name':NAME,'version':VERSION,
        'description':('MetaKat chapter pipeline with 512-page contextual page-type, side and document-boundary '
                       'inference, page-number detection, monograph title-page metadata and multi-page TOC '
                       'extraction by OpenAI GPT-6 Luna (medium reasoning) through OpenRouter with Zero Data '
                       'Retention, YOLO destination-page analysis and fuzzy chapter alignment.'),
        'definition':config,'default':False,'active':True,
    }
    provenance={'name':NAME,'version':VERSION,'status':'preparing','published':False,'paid_requests':0,
                'bases':[str(monograph),str(chapter)],'files':{}}
    # Collect only configured local model resources; unused YOLO extraction and
    # bibliography weights from the earlier chapter bundle are not copied.
    resources={}
    def collect(value,source):
        if not isinstance(value,dict):return
        for key,child in value.items():
            if isinstance(child,dict):collect(child,source)
            elif isinstance(child,str) and key.endswith(('_path','_dir')):
                relative=Path(child)
                if relative.is_absolute() or '..' in relative.parts:
                    raise ValueError('Bundle model paths must be safe and relative')
                origin=source/relative
                if not origin.exists():raise FileNotFoundError(origin)
                resources[str(relative)]=origin
    for stage in ('page_number','page_type','biblio'):
        collect(config[stage],monograph)
    collect(config['chapter'],chapter)
    root.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.'+destination.name+'.',dir=root) as temporary:
        staged=Path(temporary)/destination.name
        staged.mkdir()
        for relative,source in resources.items():
            target=staged/relative
            if source.is_dir():
                shutil.copytree(source,target,ignore=shutil.ignore_patterns('.cache','__pycache__'))
            else:
                target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        (staged/'engine_config.json').write_text(json.dumps(config,indent=2,ensure_ascii=False)+'\n')
        (staged/'engine_config.yaml').write_text(yaml.safe_dump(config,sort_keys=False,allow_unicode=True))
        for file in sorted(staged.rglob('*')):
            if file.is_file():
                if file.is_symlink():raise ValueError('Bundle must be self-contained')
                provenance['files'][str(file.relative_to(staged))]={'bytes':file.stat().st_size,'sha256':sha256(file)}
        from metakat.engine_config import prepare_engine_config
        from metakat.process_batch import _preflight_engine_requirements
        from metakat.chapter.engines.core.chapter_extraction.engine_vlm import ChapterExtractionEngineVLM
        from metakat.biblio.engines.core.definitions import load_biblio_core_engine
        resolved=prepare_engine_config(config,base_dir=staged)
        _preflight_engine_requirements(resolved)
        # Constructors read prompts/schema/config but defer all paid clients.
        biblio=load_biblio_core_engine(resolved['biblio']['core'])
        toc=ChapterExtractionEngineVLM(resolved['chapter']['core']['extraction'],client=object())
        if toc.process(()).chapters:raise AssertionError('Empty TOC must not request inference')
        if toc.vlm_config.max_attempts!=1 or not toc.vlm_config.zdr:raise AssertionError('VLM policy mismatch')
        provenance['toc_schema_sha256']=sha256(Path(__import__(toc.__module__,fromlist=['x']).__file__).parent/'vlm/schema.json')
        for item in toc.prompts:
            # Prompts are package resources, not model assets; record their exact text.
            provenance.setdefault('toc_prompt_sha256',{})[item.role]=hashlib.sha256(item.text.encode()).hexdigest()
        import metakat
        source_root=Path(metakat.__file__).resolve().parents[1]
        provenance['metakat_root']=str(source_root)
        provenance['metakat_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source_root,text=True).strip()
        if yaml.safe_load((staged/'engine_config.yaml').read_text())!=config:raise AssertionError('JSON/YAML mismatch')
        staged.rename(destination)
    descriptor_path.write_text(json.dumps(descriptor,indent=2,ensure_ascii=False)+'\n')
    provenance.update(status='ready',validation=['JSON/YAML/descriptor equivalent','all configured model resources copied',
        'pipeline engine/stage dependency preflight','monograph VLM constructor','TOC VLM constructor and empty-input no-request path'],
        limitations=['No real API inference performed; quality and cost not measured for this bundle'])
    preparation_path.write_text(json.dumps(provenance,indent=2,ensure_ascii=False)+'\n')
    print('Prepared',destination,flush=True)
    with zipfile.ZipFile(archive_path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as archive:
        for file in sorted(destination.rglob('*')):
            if file.is_file():archive.write(file,file.relative_to(destination))
    with zipfile.ZipFile(archive_path) as archive:
        if json.loads(archive.read('engine_config.json'))!=config:raise AssertionError('Archive config mismatch')
    provenance['archive']={'path':str(archive_path),'bytes':archive_path.stat().st_size,'sha256':sha256(archive_path)}
    preparation_path.write_text(json.dumps(provenance,indent=2,ensure_ascii=False)+'\n')
    print('Archive',archive_path,archive_path.stat().st_size,flush=True)


if __name__=='__main__':
    main()
