import copy

import torch

from page_sequence.loading import training_loader
from page_sequence.runner import train, load_checkpoint
from page_sequence.smoke import prepare_fixture
from page_sequence.data import WindowDataset
from page_sequence.features import FeaturePipeline
from page_sequence.index import PageIndex
from page_sequence.sampling import WindowSampler


def test_forkserver_matches_inline_sampling_and_batch_values(tmp_path):
    cfg=prepare_fixture(tmp_path, steps=4)
    index=PageIndex(cfg['index'])
    dataset=WindowDataset(index, FeaturePipeline.from_config(cfg['features']))
    loaders=[]
    try:
        all_batches=[]
        for workers in (0, 2):
            config=dict(cfg, workers=workers, worker_start_method='forkserver')
            sampler=WindowSampler(index, 12, variable_lengths=True)
            loader,batches=training_loader(dataset,sampler,config)
            loaders.append(loader)
            batches.batches=3
            all_batches.append(list(loader))
        for a,b in zip(*all_batches):
            assert a['rows'] == b['rows']
            for key in ('attention_mask','type','placement','starts','pair_mask','weight'):
                torch.testing.assert_close(a[key],b[key])
            for key in a['features']:
                torch.testing.assert_close(a['features'][key],b['features'][key])
    finally:
        for loader in loaders:
            if getattr(loader,'_iterator',None) is not None:
                loader._iterator._shutdown_workers()


def test_eight_forkserver_workers_train_and_validate(tmp_path):
    config=prepare_fixture(tmp_path,steps=4)
    config.update(workers=8,worker_start_method='forkserver',variable_lengths=True)
    if torch.cuda.is_available():
        config.update(device='cuda',precision='float16',deterministic=False)
    train(config)
    _,state=load_checkpoint(tmp_path/'run/latest.pt')
    assert state['step']==4 and state['training']['finished']
