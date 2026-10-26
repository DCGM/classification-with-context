"""Explicit ClearML scalars/configuration; checkpoints remain local without a file server."""
import json
from pathlib import Path

import numpy as np

from .schema import PAGE_TYPES, PLACEMENTS


class Tracking:
    def __init__(self, config, provenance, resume=False):
        self.task = None
        settings = config.get('clearml', {})
        if not settings.get('enabled', False):
            return
        saved = Path(config['run_dir']) / 'clearml.json'
        if resume and not saved.exists():
            raise FileNotFoundError('Cannot append ClearML history: missing run_dir/clearml.json')
        from clearml import Task
        if not settings.get('report_in_subprocess', True):
            # ClearML's default fork retains a copy of the populated catalogue.
            # Use SDK thread reporting without changing the user's global config.
            from clearml.config import config_obj, config as sdk_config
            config_obj.set_overrides({'sdk': {'development': {'report_use_subprocess': False}}})
            sdk_config.clear_config_impl()
            if bool(Task._report_subprocess_enabled):
                raise RuntimeError('ClearML thread-reporting override was not applied')
        previous = json.loads(saved.read_text()) if resume else None
        self.task = Task.init(
            project_name=settings.get('project', 'classification-with-context'),
            task_name=settings.get('name', Path(config['run_dir']).name),
            reuse_last_task_id=previous['id'] if previous else False,
            # Integer zero preserves history while disabling SDK iteration offsets.
            # The runner already reports checkpoint-restored absolute global steps.
            continue_last_task=0 if previous else False, output_uri=False,
            auto_connect_arg_parser=False, auto_connect_frameworks=False,
            auto_connect_streams=True, auto_resource_monitoring=True)
        self.task.connect_configuration(config, name='experiment')
        self.task.connect_configuration(provenance, name='provenance')
        self.logger = self.task.get_logger()
        info = {'id': self.task.id, 'url': self.task.get_output_log_web_page()}
        saved.write_text(json.dumps(info, indent=2) + '\n')
        print(json.dumps({'clearml': info}), flush=True)

    def report(self, record):
        if self.task is None:
            return
        step = record['step']
        for group in ('loss', 'valid_weight'):
            for name, value in record[group].items():
                self.logger.report_scalar(f'{record["stage"]}/{group}', name, value, step)
        for name in ('gradient_norm', 'io_seconds', 'pages_per_second', 'coverage_epochs'):
            self.logger.report_scalar('training', name, record[name], step)
        for name, value in record.get('storage_pages', {}).items():
            self.logger.report_scalar('training/storage_pages', name, value, step)
        if 'loss_scale' in record:
            self.logger.report_scalar('training', 'loss_scale', record['loss_scale'], step)
        if 'dev' in record:
            report = record['dev']
            for group in ('curated_type', 'weak_type_agreement', 'placement_db_agreement'):
                for name in ('accuracy', 'weighted_precision', 'weighted_recall', 'weighted_f1', 'macro_f1'):
                    if report[group].get(name) is not None:
                        self.logger.report_scalar(f'dev/{group}', name, report[group][name], step)
                for label, scores in report[group]['per_class'].items():
                    for name, value in scores.items():
                        self.logger.report_scalar(f'dev/{group}/{name}', label, value, step)
                labels = PLACEMENTS if group == 'placement_db_agreement' else PAGE_TYPES
                self.logger.report_confusion_matrix(
                    title=f'dev/{group}/confusion', series='counts', iteration=step,
                    matrix=np.asarray(report[group]['confusion']),
                    xaxis='Predicted', yaxis='True', xlabels=list(labels), ylabels=list(labels))
            for group, values in report.get('starts', {}).items():
                for name, value in values.items():
                    if value is not None:
                        self.logger.report_scalar('dev/starts' if group == 'all' else f'dev/starts/{group}', name, value, step)
            for group, values in report['join_detection'].items():
                self.logger.report_scalar('dev/join_recall', group, values['recall'], step)
            for name, value in report.get('side_sequence_db_agreement', {}).items():
                if value is not None:
                    self.logger.report_scalar('dev/side_sequence_db_agreement', name, value, step)
            for name, value in report.get('exact_spans', {}).items():
                if value is not None:
                    self.logger.report_scalar('dev/exact_spans', name, value, step)
            if report.get('exact_stream_fraction') is not None:
                self.logger.report_scalar('dev/segmentation', 'exact_stream_fraction', report['exact_stream_fraction'], step)
            for group, directions in report.get('front_back_confusion', {}).items():
                for direction, values in directions.items():
                    for name, value in values.items():
                        if value is not None:
                            self.logger.report_scalar(f'dev/{group}/front_back/{name}', direction, value, step)
            for direction in ('front_at_end', 'back_at_start'):
                for name, value in report.get('front_back_position', {}).get(direction, {}).items():
                    if value is not None:
                        self.logger.report_scalar(f'dev/front_back_position/{name}', direction, value, step)
            if 'contexts' in report:
                self.report_contexts(report['contexts'], step)
            self.task.flush(wait_for_uploads=True)

    def report_contexts(self, report, step, prefix='dev/context'):
        if self.task is None:
            return
        self.logger.report_scalar(prefix, 'seconds', report['seconds'], step)
        self.logger.report_scalar(prefix, 'complete', int(report['complete']), step)
        for length, scores in report['lengths'].items():
            title = f'{prefix}/{length}'
            self.logger.report_scalar(title, 'pages', scores['pages'], step)
            self.logger.report_scalar(title, 'complete', int(scores['complete']), step)
            for group in ('curated_type', 'weak_type_agreement', 'placement_db_agreement'):
                for metric in ('accuracy', 'weighted_f1', 'macro_f1'):
                    value = scores[group].get(metric)
                    if value is not None:
                        self.logger.report_scalar(title + '/' + group, metric, value, step)
            if scores['starts'] is not None:
                for metric in ('f1', 'precision', 'recall', 'n', 'positive'):
                    self.logger.report_scalar(title + '/starts', metric, scores['starts'][metric], step)

    def close(self, failed=False):
        if self.task is not None:
            if failed:
                self.task.mark_failed(status_reason='training raised an exception')
            self.task.close()
