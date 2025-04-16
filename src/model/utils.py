import wandb


class WanbSyncLogger:
    """Synchronized wandb logger. 
    The raw version's step counting is async, because of pl calls """
    def __init__(self, run, log_every_n_steps=None, step_suffix='_step', epoch_suffix='_epoch'):
        self.run = run
        self.log_every_n_steps = log_every_n_steps
        self.cached_data = dict()
        self.current_step = 1
        self.old_current_epoch = 1
        self.current_epoch = 1
        
        self._step_suffix = step_suffix
        self._epoch_suffix = epoch_suffix

    def add_step(self):
        self.current_step += 1

    def add_epoch(self):
        self.current_epoch += 1

    def sync_epoch(self):
        self.old_current_epoch = self.current_epoch

    def log(self, data: dict, on_step=None, on_epoch=None, step=None):
        if not self.log_every_n_steps or (on_step is None and on_epoch is None):
            wandb.log(data, step=step)
            return None
        
        # if k in d, append v， else create [v]
        _dict_list_updater = lambda d, k, v: d[k].append(v) if k in d.keys() else d.update({k: [v]})
        # reduction by the `mean` of a list
        _list_reducer = lambda x: sum(x) / len(x)

        # on-step metrics
        if on_step:
            # 1. update
            for k, v in data.items():
                _dict_list_updater(self.cached_data, k + self._step_suffix, v)
            
            # 2. log & 3. reset
            if self.current_step % self.log_every_n_steps == 0:
                wandb.log({k + self._step_suffix: _list_reducer(self.cached_data[k + self._step_suffix])
                            for k in data.keys()}, step=step)
                for k in data.keys():
                    self.cached_data[k + self._step_suffix] = []

        if on_epoch:
            # on-epoch metrics
            if self.current_epoch != self.old_current_epoch:
                epoch_keys = list(filter(lambda x: self._epoch_suffix in x, self.cached_data.keys()))
                wandb.log({k: _list_reducer(self.cached_data[k]) for k in epoch_keys}, step=step)
                for k in epoch_keys:
                    self.cached_data[k] = []
            else:
                for k, v in data.items():
                    _dict_list_updater(self.cached_data, k + self._epoch_suffix, v)