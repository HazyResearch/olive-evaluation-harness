from typing import Any, List
import inspect

import torch
import hydra
from flash_attn.losses.cross_entropy import CrossEntropyLoss
from pytorch_lightning import LightningModule, LightningDataModule
from torchmetrics import MetricCollection

from einops import rearrange

from omegaconf import OmegaConf

from train.utils.utils import get_logger
from train.optim.param_grouping import group_parameters_for_optimizer
from train.utils.checkpoint import load_checkpoint
from train.config import Config

logger = get_logger(__name__)


class SequenceModel(LightningModule):

    def __init__(
        self, 
        config: Config, 
        model: torch.nn.Module,
        datamodule: LightningDataModule,
    ):
        """If model_cfg is passed, it will take precedence over cfg.model
        """
        super().__init__()      
        self.config = config
        self.model = model
        self.datamodule = datamodule
        
        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters(config.to_dict())

        # instantiate loss
        self.loss_fn = CrossEntropyLoss(inplace_backward=True)
        self.loss_fn_val = CrossEntropyLoss(inplace_backward=True)
        self.set_metrics()
        
    def set_metrics(self):
        # instantiate metrics
        from train.metrics.perplexity import Perplexity
        from train.metrics.num_tokens import NumTokens
        metrics = MetricCollection({
            "ppl": Perplexity(),
            "num_tokens": NumTokens()
        })
        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_metrics = metrics.clone(prefix='train/')
        self.val_metrics = metrics.clone(prefix='val/')
        self.test_metrics = metrics.clone(prefix='test/')
        
    def instantiate_model(self):
        from transformers import AutoModelForCausalLM
        self.model = AutoModelForCausalLM.from_pretrained("google/gemma-2b-it")
        self.model = self.model.to(self.device)

    def warmstart(self):
        if self.config.train.get('warmstart', None) is not None:
            logger.info(f"Warm-starting with weights from {self.config.train.warmstart.path}")
            state_dict = load_checkpoint(self.config.train.warmstart.path)
            if self.config.train.warmstart.get('post_process', None) is not None:
                state_dict = hydra.utils.instantiate(self.config.train.warmstart.post_process,
                                                     state_dict)
            load_return = self.model.load_state_dict(state_dict, strict=False)
            logger.info(load_return)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def step(self, batch: Any, is_train=True):
        try:
            x, y, lengths = batch
        except ValueError:
            x, y = batch
            lengths = None
        output = self.forward(x) if lengths is None else self.forward(x, lengths=lengths)
        loss = self.loss_fn(output, y) if is_train else self.loss_fn_val(output, y)
        return loss, output, y

    def shared_step(self, batch: Any, batch_idx: int, phase='train'):
        loss, output, targets = self.step(batch, is_train=(phase == 'train'))
        metrics = getattr(self, f'{phase}_metrics')
        metrics(output, targets)
        log_on_step = phase == 'train'
        self.log(f"{phase}/loss", loss, on_step=log_on_step, on_epoch=True,
                 prog_bar=False, sync_dist=True)
        # https://pytorch-lightning.readthedocs.io/en/stable/visualize/logging_advanced.html#enable-metrics-for-distributed-training
        # We need to log the Metrics object, not the metric result, since otherwise
        # pytorch-lightning will use torch.mean to reduce it.
        # This would be wrong for perplexity, for example.
        self.log_dict(metrics, on_step=log_on_step, on_epoch=True, prog_bar=True, sync_dist=True)
        return {"loss": loss, "output": output, "targets": targets}

    def training_step(self, batch: Any, batch_idx: int):
        return self.shared_step(batch, batch_idx, phase='train')

    def validation_step(self, batch: Any, batch_idx: int):
        return self.shared_step(batch, batch_idx, phase='val')

    def test_step(self, batch: Any, batch_idx: int):
        return self.shared_step(batch, batch_idx, phase='test')

    def configure_optimizers(self):
        if self.config.optimizer_groups is not None:  # Set zero weight decay for some params
            parameters = group_parameters_for_optimizer(
                self.model, 
                self.config.optimizer,
                bias_weight_decay=self.config.optimizer_groups.bias_weight_decay,
                normalization_weight_decay=self.config.optimizer_groups.normalization_weight_decay
            )
        else:
            # parameters = self.model.parameters()
            parameters = self.parameters() # [21-09-08] AG: this will train task specific parameters such as Retrieval head for AAN
        optimizer = self.config.optimizer.instantiate(parameters)

        # Log optimizer info
        for i, g in enumerate(optimizer.param_groups):
            ntensors = len(g['params'])
            nparams = sum(p.numel() for p in g['params'])
            hparams = {k: v for k, v in g.items() if k != 'params'}
            print(f'Optimizer group {i}: {ntensors} tensors, {nparams} parameters, {hparams}')
        
        # load a pretrained optimizer state
        if self.config.pretrained_optimizer_path is not None:
            print(f"Loading optimizer state from {self.config.pretrained_optimizer_path}")
            state_dict = torch.load(self.config.pretrained_optimizer_path)

            if "pytorch-lightning_version" in state_dict:
                # this is a pytorch-lightning module checkpoint, so we need to extract
                # the optimizer state from the checkpoint
                if len(state_dict['optimizer_states']) > 1:
                    raise ValueError("Multiple optimizers are not currently supported")
                
                state_dict = state_dict['optimizer_states'][0]

            optimizer.load_state_dict(state_dict)

        if self.config.scheduler is None:
            return optimizer
        
        # lr_scheduler should be called either every step (default) or every epoch
        lr_scheduler = self.config.scheduler.instantiate(optimizer)
        return [optimizer], {'scheduler': lr_scheduler,
                             'interval': self.config.get('scheduler_interval', 'step'),
                             'monitor': self.config.get('scheduler_monitor', 'val/loss')}

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        # https://pytorch-lightning.readthedocs.io/en/latest/guides/speed.html#set-grads-to-none
        # TD [2022-04-30]: DeepSpeed optimizer uses the kwarg set_grad_to_none instead of set_to_none
        if 'set_to_none' in inspect.signature(optimizer.zero_grad).parameters:
            optimizer.zero_grad(set_to_none=True)
        else:
            optimizer.zero_grad()

    def on_save_checkpoint(self, checkpoint):
        # TD [2022-08-07] ['epoch_loop.batch_progress']['total']['completed'] is 1 iteration
        # behind, so we're using the optimizer's progress.
        checkpoint['loops']['fit_loop']['epoch_loop.batch_progress']['total']['completed'] = checkpoint['loops']['fit_loop']['epoch_loop.batch_loop.optimizer_loop.optim_progress']['optimizer']['step']['total']['completed'] * self.trainer.accumulate_grad_batches
        checkpoint['loops']['fit_loop']['epoch_loop.batch_progress']['current']['completed'] = checkpoint['loops']['fit_loop']['epoch_loop.batch_loop.optimizer_loop.optim_progress']['optimizer']['step']['current']['completed'] * self.trainer.accumulate_grad_batches
        # _batches_that_stepped tracks the number of global steps, not the number
        # of local steps, so we don't multiply with self.trainer.accumulate_grad_batches here.
        checkpoint['loops']['fit_loop']['epoch_loop.state_dict']['_batches_that_stepped'] = checkpoint['loops']['fit_loop']['epoch_loop.batch_loop.optimizer_loop.optim_progress']['optimizer']['step']['total']['completed']


class SequenceLMModel(SequenceModel):

    def step(self, batch: Any, is_train=True):
        if len(batch) == 3:
            x, y, _ = batch
        else:
            x, y = batch
        output = self.forward(x).logits
        output = rearrange(output, '... C -> (...) C')
        y = rearrange(y, '... -> (...)')
        loss = self.loss_fn(output, y) if is_train else self.loss_fn_val(output, y)
        return loss, output, y

    def shared_step(self, batch: Any, batch_idx: int, phase='train'):
        loss, output, targets = self.step(batch, is_train=(phase == 'train'))
        # Passing the loss to the perplexity metrics to avoid recomputation
        metrics = getattr(self, f'{phase}_metrics')
        metrics(output, targets, loss=loss)
        log_on_step = phase == 'train'
        # print(f"{batch_idx=},{loss=},{self.trainer.global_step=}")
        self.log(f"{phase}/loss", loss, on_step=log_on_step, on_epoch=True,
                 prog_bar=False, sync_dist=True)
        # https://pytorch-lightning.readthedocs.io/en/stable/visualize/logging_advanced.html#enable-metrics-for-distributed-training
        # We need to log the Metrics object, not the metric result, since otherwise
        # pytorch-lightning will use torch.mean to reduce it.
        # This would be wrong for perplexity, for example.
        self.log_dict(metrics, on_step=log_on_step, on_epoch=True, prog_bar=True, sync_dist=True)
        return {"loss": loss, "output": output, "targets": targets}



