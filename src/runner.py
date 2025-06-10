import logging
import threading
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from src.data.dataset import FSSDataset
from src.loss.utils import MultipleLosses
from src.metric.utils import SuperMetricCollection

from src.model import REGISTERED_MODELS


log = logging.getLogger(__name__)


def setup_dataset(config: DictConfig) -> FSSDataset:
    # convert DictConfig to dict, since transform is not a primitive type. 
    config.runner.dataset.n_shot = config.exp.n_shot
    data_config = OmegaConf.to_container(config.runner.dataset, resolve=True)
    transform = instantiate(config.runner.aug)
    data_config.update(transform)
    if 'also_val_at' in data_config.keys() and data_config['also_val_at'] is not None:
        data_config['also_val_at']['n_shot'] = config.exp.n_shot
        data_config['also_val_at'].update(transform)
        
    return FSSDataset(data_config=data_config, loader_config=config.runner.loader)

def setup_model(config: DictConfig):
    loss_func = MultipleLosses(instantiate(config.runner.loss))
    metric = SuperMetricCollection(
        {metric_name: metric for metric_name, metric in instantiate(config.runner.metric).items()})
    checkpointing = lambda _: instantiate(config.checkpoint)  # delayed init at rank zero only, for DDP.

    model_class = REGISTERED_MODELS.get(config.runner.model.name)
    if model_class is not None:
        model = model_class(
            loss_func=loss_func,
            metrics=metric,
            model_cfg=config.runner.model,
            optimizer_cfg=config.runner.optimizer,
            scheduler_cfg=config.runner.scheduler,
            cfg=config,
            checkpointing=checkpointing
        )
    else: raise NameError(f'Unknown model name: {config.runner.model.name}')

    return model


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



