import logging
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

import lightning.pytorch as pl
from lightning_fabric.utilities.rank_zero import _get_rank
from lightning.pytorch.callbacks.callback import Callback
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from lightning.pytorch.utilities.rank_zero import rank_prefixed_message, rank_zero_warn

log = logging.getLogger(__name__)

CR = '\033[93m'
CB = '\033[94m'
CG = '\033[92m'
CE = '\033[0m'


class FullEarlyStopping(Callback):
    mode_dict = {"min": torch.lt, "max": torch.gt}

    def __init__(
        self,
        monitor: str,
        min_delta: float = 0.0,
        consecutive: int = 3,
        overall: int = 15,
        mode: str = 'min',
    ):
        super().__init__()
        self.monitor = monitor
        self.min_delta = min_delta
        self.ordered = consecutive
        self.overall = overall
        self.wait_count = 0
        self.best_count = 0
        self.stopped_epoch = 0
        self.mode = mode

        if mode == 'min':
            starting = np.Inf
        else:
            starting = -1 * np.Inf

        self.previous = torch.tensor(starting)
        self.best = torch.tensor(starting)

    @property
    def state_key(self) -> str:
        return self._generate_state_key(monitor=self.monitor, mode=self.mode)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "wait_count": self.wait_count,
            "stopped_epoch": self.stopped_epoch,
            "previous": self.previous,
            "best": self.best,
            "ordered": self.ordered,
            "overall": self.overall,
            "mode": self.mode
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.wait_count = state_dict["wait_count"]
        self.stopped_epoch = state_dict["stopped_epoch"]
        self.previous = state_dict["previous"]
        self.best = state_dict["best"]
        self.ordered = state_dict["ordered"]
        self.overall = state_dict["overall"]
        self.mode = state_dict["mode"]

    def _should_skip_check(self, trainer: "pl.Trainer") -> bool:
        from lightning.pytorch.trainer.states import TrainerFn
        return trainer.state.fn != TrainerFn.FITTING or trainer.sanity_checking

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        pass

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self._should_skip_check(trainer):
            return
        self._run_early_stopping_check(trainer)

    def _run_early_stopping_check(self, trainer: "pl.Trainer") -> None:
        """Checks whether the early stopping condition is met and if so tells the trainer to stop the training."""
        logs = trainer.callback_metrics

        current = logs[self.monitor].squeeze()
        should_stop = self._evaluate_stopping_criteria(current)

        # stop every ddp process if any world process decides to stop
        should_stop = trainer.strategy.reduce_boolean_decision(should_stop, all=False)
        trainer.should_stop = trainer.should_stop or should_stop
        if should_stop:
            self.stopped_epoch = trainer.current_epoch

    def _evaluate_stopping_criteria(self, current: Tensor) -> bool:
        should_stop = False
        
        # Most recent better value for the metric.
        lt_cond = current - self.min_delta < self.previous.to(current.device)
        gt_cond = current - self.min_delta > self.previous.to(current.device)
        if (lt_cond and self.mode == 'min') or (gt_cond and self.mode == 'max'):
            should_stop = False
            if torch.isfinite(self.previous):
                print(CG + f"\nMetric {self.monitor} improved by {abs(self.previous - current):.3f} >="
                      f" min_delta = {abs(self.min_delta)}." + CE)
                self.wait_count = 0
        else:
            self.wait_count += 1
            print(CB + f"\nEpochs since last improvement in {self.monitor} value: {self.wait_count}." + CE)
            if self.wait_count >= self.ordered:
                should_stop = True
                print(CR + f"\nMonitored metric {self.monitor} did not improve in the last {self.wait_count} records." + CE)
        self.previous = current

        # Overall best value for the metric
        lt_cond = current - self.min_delta < self.best.to(current.device)
        gt_cond = current - self.min_delta > self.best.to(current.device)
        if (lt_cond and self.mode == 'min') or (gt_cond and self.mode == 'max'):
            should_stop = False
            if torch.isfinite(self.best):
                print(CG + f"New best score: {current:.3f}" + CE)
            else:
                print(CG + f"Metric {self.monitor} improved. New best score: {current:.3f}" + CE)
            self.best_count = 0
            self.best = current
        else:
            self.best_count += 1
            print(CB + f"\nEpochs since best {self.monitor} value: {self.best_count}." + CE)
            if self.best_count >= self.overall:
                should_stop = True
                print(CR + f"\nMonitored metric {self.monitor} did not improve in the last {self.best_count} records." + CE)

        return should_stop