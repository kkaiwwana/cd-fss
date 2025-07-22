def quick_compose(cfg_path='../../config', cfg_name='main.yaml', overrides=None, verbose=True):
    from hydra import initialize, compose
    from omegaconf import OmegaConf
    
    with initialize(version_base=None, config_path=cfg_path):
        cfg = compose(config_name=cfg_name, overrides=overrides)
        if verbose:
            print(OmegaConf.to_yaml(cfg))
    
    return cfg


if __name__ == '__main__':
    print('not a script. for notebook only.')