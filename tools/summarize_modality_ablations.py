"""Summarize completed, selected fine-tuning checkpoints without loading Torch."""
import argparse
import csv
import json
from pathlib import Path


def selected_validation(run):
    selected = None
    for line in (run / 'validation.jsonl').open():
        value = json.loads(line)
        if value['stage'] != 'finetune':
            continue
        if selected is None or value['dev']['curated_type']['weighted_f1'] > selected['dev']['curated_type']['weighted_f1']:
            selected = value
    if selected is None:
        raise ValueError(f'no fine-tuning validation in {run}')
    return selected


def summarize(root):
    settings = json.loads((root / 'monitor-config.json').read_text())
    state = json.loads((root / 'state.json').read_text())
    if not state['all_finished']:
        raise ValueError('queue is not complete')
    results = []
    for job in settings['jobs']:
        config = json.loads((Path(job['root']) / 'config.json').read_text())
        run = Path(config['run_dir'])
        for name in ('finetune_best.pt', 'pretrain_final.pt', 'finetune_init.pt', 'finetune_final.pt'):
            if not (run / name).exists():
                raise ValueError(f'missing retained checkpoint {run / name}')
        selected = selected_validation(run)
        results.append(dict(id=job['id'], slot=job.get('slot'),
                            enabled_modalities=config['enabled_modalities'], run_dir=str(run),
                            selected_step=selected['step'], dev=selected['dev'],
                            clearml=json.loads((run / 'clearml.json').read_text())))
    report = dict(complete=True, selection='development curated page-type weighted F1',
                  experiments=results)
    (root / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
    with (root / 'results.csv').open('w', newline='') as stream:
        fields = ['id', 'slot', 'modalities', 'selected_step', 'type_f1', 'type_macro_f1',
                  'side_f1', 'boundary_f1', 'boundary_fp', 'boundary_fn',
                  'provided_type_f1', 'provided_boundary_f1']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for r in results:
            d, provided = r['dev'], r['dev']['metadata_provided']
            boundary = d['starts']['all']
            writer.writerow(dict(id=r['id'], slot=r['slot'], modalities='+'.join(r['enabled_modalities']),
                                 selected_step=r['selected_step'], type_f1=d['curated_type']['weighted_f1'],
                                 type_macro_f1=d['curated_type']['macro_f1'],
                                 side_f1=d['placement_db_agreement']['weighted_f1'],
                                 boundary_f1=boundary['f1'], boundary_fp=boundary['fp'], boundary_fn=boundary['fn'],
                                 provided_type_f1=provided['curated_type']['weighted_f1'],
                                 provided_boundary_f1=provided['starts']['all']['f1']))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    summarize(parser.parse_args().root)
