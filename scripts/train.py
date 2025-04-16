import sys
import hydra
import torch
import wandb
import logging
import threading
import pytorch_lightning as pl

from pathlib import Path
from datetime import datetime
from omegaconf import DictConfig
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

log = logging.getLogger(__name__)


CONFIG_PATH = str(Path.cwd() / 'config')
CONFIG_NAME = 'train'
WANDB_API_KEY = open('wandb_api_key.txt', mode='r').read()


def timeout_input(timeout=5):
    result = [None]

    def get_input():
        result[0] = int(input())

    thread = threading.Thread(target=get_input)
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        log.info("Time out. Select the last one by default.")
        thread.join(0)
    return result[0]


def try_resume_training(experiment):
    """specify an existing experiment name, them try resume training"""
    save_dir = Path(experiment.save_dir).resolve().joinpath('../')  # back to top folder
    checkpoints = list(save_dir.glob(f'**/{experiment.uuid.split('@')[0]}*/checkpoints/*.ckpt'))
    
    log.info(f'Searching {save_dir}.')

    if not checkpoints:
        return None
    
    log.info('\nFound following check points:\n\n')
    for i, ckpt in enumerate(checkpoints):
        log.info(f'{i} - {ckpt}\n')
    log.info(f'Please select one checkpoint by its number.')
    
    ckpt_idx = timeout_input(timeout=10)
    ckpt_path = checkpoints[-1] if not ckpt_idx else checkpoints[ckpt_idx]
    log.info(f'Use checkpoint: {ckpt_path}')
    
    return ckpt_path


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(config):
    sys.path.append('./')
    from src.runner import setup_model, setup_dataset
    from src.callback.git_diff import GitDiffCallback
    
    wandb.login(key=WANDB_API_KEY)
    pl.seed_everything(config.experiment.seed)
    ckpt_path = try_resume_training(config.experiment)
    
    Path(config.experiment.save_dir).mkdir(exist_ok=True, parents=True)

    dataset = setup_dataset(config)
    
    # terrible code:( have to compromise.
    config.scheduler.scheduler_params.steps_per_epoch = len(dataset.train_dataloader())
    model = setup_model(config)
        
    if ckpt_path is None:
        run_id = config.experiment.uuid 
    else:
        model.load_state_dict(torch.load(ckpt_path, weights_only=True), strict=True)
        run_id = config.experiment.uuid + f'@RESUME@{datetime.now().strftime("%m%d_%H%M%S")}'
    
    logger = WandbLogger(
        project=config.experiment.project,
        save_dir=config.experiment.save_dir,
        id= run_id,
        log_model=False,
    )
    
    # make training amount invariant to epoch splits
    if 'train_epoch_splits' in config.loader.keys() and isinstance(config.loader.train_epoch_splits, int):
        config.trainer.max_epoch *= config.loader.train_epoch_splits
    
    callbacks = [GitDiffCallback(config), ModelCheckpoint(save_top_k=0)]
    # start training
    trainer = pl.Trainer(logger=logger, callbacks=callbacks, **config.trainer)
    trainer.fit(model, datamodule=dataset)
    
    
if __name__ == '__main__':
    sys.path.append('./')
    
    main()
