import importlib
from typing import List, Optional, Sequence
from pathlib import Path
import re

import click
from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy

from train.callbacks.prediction_log import PredictionLogger
from train.utils import utils
from train.utils.utils import import_object
from train.config import Config, BaseConfig

log = utils.get_logger(__name__)

import torch
import torch.distributed
import pandas as pd
import tempfile
import os
import subprocess


def last_modification_time(path):
    """Including files / directory 1-level below the path
    """
    path = Path(path)
    if path.is_file():
        return path.stat().st_mtime
    elif path.is_dir():
        return max(child.stat().st_mtime for child in path.iterdir())
    else:
        return None


def train(config: Config) -> Optional[float]:
    """Contains training pipeline.
    Instantiates all PyTorch Lightning objects from config.

    Args:
        config (DictConfig): Configuration composed by Hydra.

    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """
    # compute correct batch_sizes and gradient accumulation
    assert config.global_batch_size % config.trainer.devices == 0
    assert config.global_batch_size % config.datamodule.batch_size == 0
    config.trainer.accumulate_grad_batches = config.global_batch_size // config.datamodule.batch_size // config.trainer.devices

    # add the name to the dirpath and to the wandb logger
    config.checkpointer.dirpath = os.path.join(config.checkpointer.dirpath, config.name)
    config.logger.name = config.name
    
    if (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0:
        config.print()
    
    # Set seed for random number generators in pytorch, numpy and python.random
    seed_everything(config.seed, workers=True)


    # Initialize and prepare data 
    datamodule: LightningDataModule = config.datamodule.instantiate()
    datamodule.prepare_data()
    datamodule.setup()

    model = config.model.instantiate()

    task: LightningModule = import_object(config.task)(
        config=config, model=model, datamodule=datamodule
    )

    callbacks: List[Callback] = [c.instantiate() for c in config.callbacks]
    
    # SE (02-22-24): Checkpointer is technically a callback, but we grant it special
    # treatment because it should never be excluded 
    checkpointer: ModelCheckpoint = config.checkpointer.instantiate()
    callbacks.append(checkpointer)
    
    # Init lightning loggers
    logger: WandbLogger = config.logger.instantiate()

    pred_logger = PredictionLogger(wandb_logger=logger, tokenizer_name=config.datamodule.tokenizer_name)
    callbacks.append(pred_logger)
    
    # optionally freeze some layers
    if config.model.trainable_params is not None:
        frozen, trainable = [], []
        for name, param in model.named_parameters():
            if not re.match(config.model.trainable_params, name):
                param.requires_grad = False
                frozen.append(name)
            else:
                trainable.append(name)
        print(f"Frozen: {frozen}")
        print(f"Trainable: {trainable}")
        print(f"Count Frozen: {len(frozen)}, Count Trainable: {len(trainable)}")
    

    # only log hyperparameters and parameter counts if on rank 0 in torch lightning
    if (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0:
        print("Logging config, params, and code...")
        logger.log_hyperparams(config.to_dict())
        logger.log_metrics(
             {
                 "params/count": sum(param.numel() for param in model.parameters()),
                 "params/count_trainable": sum(param.numel() for param in model.parameters() if param.requires_grad),
                 "params/count_frozen": sum(param.numel() for param in model.parameters() if not param.requires_grad),
             }
        )
        logger.experiment.log_code(
            root=os.path.dirname(os.path.dirname(__file__)),
            name="olive",
            include_fn=lambda path, root: path.endswith(".py")
        )
        print("done.")

    # select the right checkpoint to resume from 
    # if config.resume_from_checkpoint is not None and config.resume, the latter takes 
    # precedence
    fit_kwargs = {}
    if config.resume_from_path is not None:
        fit_kwargs['ckpt_path'] = config.resume_from_path
    
    if config.resume_from_self:
        checkpoint_path = Path(checkpointer.dirpath)
        if checkpoint_path.is_dir():
            checkpoint_path = checkpoint_path / 'last.ckpt'
            autosave_ckpt = checkpoint_path / '.pl_auto_save.ckpt'

            # if autosave_ckpt is newer than last_ckpt, use it
            if ((not checkpoint_path.exists())
                or (autosave_ckpt.exists()
                    and last_modification_time(autosave_ckpt) > last_modification_time(checkpoint_path))):
                checkpoint_path = autosave_ckpt
 
            if not checkpoint_path.exists():
                log.warn("`resume=True` but no checkpoint found, starting from scratch.")
            else:
                fit_kwargs['ckpt_path'] = str(checkpoint_path)
    
    
    # load the `pretrained_model_path`` if it is passed and fit_kwargs is not set
    # we wait until here to load the state dict to avoid unecessary loading in 
    # case of resuming from a checkpoint
    # `pretrained_optimizer_path` is loaded later within the `configure_optimizers` 
    # method
    if config.pretrained_model_path is not None and 'ckpt_path' not in fit_kwargs:
        print(f"Loading pretrained model from {config.pretrained_model_path}")
        state_dict = torch.load(config.pretrained_model_path)
        if "pytorch-lightning_version" in state_dict:
            # this is a pytorch-lightning module checkpoint, so we need to extract
            # the module state from the checkpoint
            state_dict = state_dict["state_dict"]
        task.load_state_dict(state_dict)


    # configure ddp automatically
    devices = config.trainer.devices
    n_devices = len(devices) if isinstance(devices, Sequence) else devices  # trainer.devices could be [1, 3] for example
    if n_devices > 1:
        assert config.trainer.strategy == "ddp"
        strategy = DDPStrategy(
            find_unused_parameters=False,    # FLAG
            gradient_as_bucket_view=True,  # https://lightning.ai/docs/pytorch/stable/advanced/ddp_optimizations.html
        )
    else: 
        strategy = config.trainer.strategy


    # initialize lightning trainer
    trainer: Trainer = config.trainer.instantiate(
        callbacks=callbacks, logger=[logger], strategy=strategy
    )

    # Train the model
    if not config.test_only:
        log.info("Starting training!")
        trainer.fit(model=task, datamodule=datamodule, **fit_kwargs)

    # Evaluate model on test set, using the best model achieved during training
    if config.test_after_training:
        log.info("Starting testing!")
        trainer.test(model=task, datamodule=datamodule)


    # Make sure everything closed properly
    log.info("Finalizing!")
    utils.finish(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # Return metric score for hyperparameter optimization
    if config.optimized_metric:
        return trainer.callback_metrics[config.optimized_metric]

@click.command(help="This is your command tool. 'config_path' is the path to a python file that defines a `config` variable.")
@click.argument("config_path", type=Path)
@click.option("--updates", "-u", type=str, multiple=True, help="Update config with these key=value pairs.")
def main(
    config_path: Optional[Path],
    updates: List[str],
):
    # Load the given Python file as a module
    spec = importlib.util.spec_from_file_location("config_module", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    config: Config = config_module.config
    
    
    # perform nested updates
    # SE (02/28): We need to use `model_copy`
    for update in updates:
        arg_path, value = update.split("=")

        child, parent, relation = config, None, None
        for key in arg_path.split("."):
            next_node = getattr(child, key)
            if isinstance(next_node, BaseConfig):
                parent = child
                child = next_node
                relation = key
        if parent is None:
            child = child.model_validate({**child.to_dict(), key: value})
        else:
            setattr(parent, relation, child.model_validate({**child.to_dict(), key: value}))

    train(config=config)

if __name__ == "__main__":
    print("Done importing!")
    main()