
import wandb
import pandas as pd 
from pytorch_lightning import Callback
from pytorch_lightning.loggers import WandbLogger
from transformers import AutoTokenizer

class PredictionLogger(Callback): 

    def __init__(
        self, 
        wandb_logger: WandbLogger,
        tokenizer_name: str = "google/gemma-2b-it"
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        self.wandb_logger = wandb_logger

         
    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Called when the validation batch ends."""
 
        # `outputs` comes from `LightningModule.validation_step`
        # which corresponds to our model predictions in this case
        
        # Let's log 20 sample image predictions from first batch
        if batch_idx == 0:
            n = 20
            inputs, targets = batch[0], batch[1]

            logits = outputs["output"]
            preds = logits.argmax(-1)
                        
            df = pd.DataFrame(
                {
                    # SE(03/05): Need to flatten the targets to remove the batch dimension
                    "target_id": targets.flatten().tolist(), 
                    "pred_id": preds.tolist(),
                    "target_text": [self.tokenizer.decode(token) for token in targets.flatten()],
                    "pred_text": [self.tokenizer.decode(token) for token in preds]
                }
            )
            self.wandb_logger.log_table(key='sample_table', dataframe=df)


