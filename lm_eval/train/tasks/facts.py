from typing import Any, Dict, List
import inspect

import torch
import hydra
from flash_attn.losses.cross_entropy import CrossEntropyLoss
from pytorch_lightning import LightningModule, LightningDataModule
from torchmetrics import MetricCollection, Metric

from einops import rearrange

from omegaconf import OmegaConf

from train.utils.utils import get_logger
from train.tasks.language_model import LanguageModel
from train.optim.param_grouping import group_parameters_for_optimizer
from train.utils.checkpoint import load_checkpoint
from train.config import Config

logger = get_logger(__name__)



class FactRecall(Metric):
    r"""
    """
    is_differentiable = False
    higher_is_better = True
    full_state_update = False
    correct: torch.Tensor
    count: torch.Tensor

    def __init__(self, **kwargs: Dict[str, Any]):
        super().__init__(**kwargs)
        self.add_state("correct", default=torch.tensor(0.0, dtype=torch.float64),
                       dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0, dtype=torch.int64), dist_reduce_fx="sum")


    def update(self, preds: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, example_idxs: torch.Tensor) -> None:  # type: ignore
        """
        Compute and track per-example accuracy.

        Given a batch of predictions and targets, this function computes per-example accuracy. 
        It first filters out irrelevant data points using a mask, then calculates the 
        accuracy for each example in the batch by noting whether all predictions for 
        a given example are correct or not. 
        
        This calculated accuracy is then tracked over successive calls to the function 
        by updating an internal counter of total checked examples and the number of those 
        that are entirely correctly predicted.

        Args:
            preds (torch.Tensor): Tensor of predicted batch labels arising from the model. 
                                  The dimension should match the target tensor.
                                  
            target (torch.Tensor): Tensor of target batch labels in the data. The dimension should 
                                   match the prediction tensor.
                                  
            mask (torch.Tensor): Bool tensor that filters out predictions as well as target values 
                                 from consideration. The dimension should match the prediction and 
                                 target tensors.
                                  
            example_idxs (torch.Tensor): Tensor indicating the corresponding example index for each 
                                         prediction and target pair in the batch.
        """

        mask = mask.flatten() == 1
        preds = preds[mask]
        target = target[mask]
        example_idxs = example_idxs.flatten()[mask]

        preds = preds.argmax(dim=-1)
        correct = (preds == target).float()

        # perform a group-by example_idxs, sum the correct predictions, and compare to 
        # the number of tokens in the example
        labels = example_idxs
        values = correct

        unique_labels, labels_mapped = labels.unique(return_inverse=True)
        num_correct_by_example = torch.zeros_like(unique_labels).to(dtype=values.dtype)
        num_correct_by_example.scatter_add_(0, labels_mapped, values)
        size_by_example = torch.bincount(labels_mapped, minlength=unique_labels.size(0))
        
        correct_examples = (num_correct_by_example == size_by_example).sum()
        num_examples = len(unique_labels)

        self.count += num_examples
        self.correct += correct_examples

    def compute(self) -> torch.Tensor:
        return self.correct / self.count


class MicroFactRecall(Metric):
    is_differentiable = False
    higher_is_better = True
    full_state_update = False
    correct: torch.Tensor
    count: torch.Tensor

    def __init__(self, **kwargs: Dict[str, Any]):
        super().__init__(**kwargs)
        self.add_state("correct", default=torch.tensor(0.0, dtype=torch.float64),
                       dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0, dtype=torch.int64), dist_reduce_fx="sum")


    def update(self, preds: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> None:  # type: ignore
        preds = preds.argmax(dim=-1)
        correct = (preds == target).float()
        self.count += mask.sum()
        self.correct += correct[mask.flatten() == 1].sum()

    def compute(self) -> torch.Tensor:
        return self.correct / self.count



class FactsModel(LanguageModel):

    def set_metrics(self):
        from train.metrics.perplexity import Perplexity
        from train.metrics.num_tokens import NumTokens
        metrics = MetricCollection({
            "ppl": Perplexity(),
            "num_tokens": NumTokens(),
            "fact_recall": FactRecall(),
            "micro_fact_recall": MicroFactRecall(),
        })
        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_metrics = metrics.clone(prefix='train/')
        self.val_metrics = metrics.clone(prefix='val/')
        self.test_metrics = metrics.clone(prefix='test/')

    def step(self, batch: Any, is_train=True):
        x, y, mask, example_idxs = batch

        output = self.forward(x).logits
        output = rearrange(output, '... C -> (...) C')
        y = rearrange(y, '... -> (...)')
        
        y_masked = y.clone()
        y_masked[mask.flatten() == 0] = -100
        
        loss = self.loss_fn(output, y_masked) if is_train else self.loss_fn_val(output, y_masked)
        return loss, output, y, mask, example_idxs

    def shared_step(self, batch: Any, batch_idx: int, phase='train'):
        loss, output, targets, mask, example_idxs = self.step(batch, is_train=(phase == 'train'))
        # Passing the loss to the perplexity metrics to avoid recomputation
        metrics = getattr(self, f'{phase}_metrics')
        metrics(output, targets, loss=loss, mask=mask, example_idxs=example_idxs)
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



