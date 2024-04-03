from typing import Any

from einops import rearrange

from train.utils.utils import get_logger
from train.tasks.seq import SequenceModel

logger = get_logger(__name__)

class LanguageModel(SequenceModel):

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



