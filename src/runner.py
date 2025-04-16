from hydra.utils import instantiate
from omegaconf import DictConfig, open_dict, OmegaConf

from src.data.dataset import FSSDataset
from src.loss.utils import MultipleLosses
from src.metric.utils import SuperMetricCollection

from src.model import MatchingNetIFA


def setup_dataset(config: DictConfig) -> FSSDataset:
    # convert DictConfig to dict, since transform is not a primitive type. 
    config.dataset.n_shot = config.experiment.n_shot
    data_config = OmegaConf.to_container(config.dataset, resolve=True)
    transform = instantiate(config.aug)
    data_config.update(transform)
    if 'also_val_at' in data_config.keys() and data_config['also_val_at'] is not None:
        data_config['also_val_at']['n_shot'] = config.experiment.n_shot
        data_config['also_val_at'].update(transform)
    
    return FSSDataset(data_config=data_config, loader_config=config.loader)

def setup_model(config: DictConfig):
    loss_func = MultipleLosses(instantiate(config.loss))
    metric = SuperMetricCollection(
        {metric_name: metric for metric_name, metric in instantiate(config.metric).items()})
    checkpointing = instantiate(config.checkpoint)
    if config.model.name == 'ifa_matching':
        
        model = MatchingNetIFA(
            loss_func=loss_func,
            metrics=metric,
            model_cfg=config.model,
            optimizer_cfg=config.optimizer,
            scheduler_cfg=config.scheduler,
            cfg=config,
            checkpointing=checkpointing
        )
    else:
        raise NotImplementedError

    return model