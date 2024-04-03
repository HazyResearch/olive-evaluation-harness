from __future__ import annotations
from typing import List, Optional, Union
from pathlib import Path

import torch
from torch import nn
import os

import hydra
from omegaconf import OmegaConf, DictConfig
from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)

import sys
import torch 

from train.config import Config

def load_model(
    run_id: str,
    device: Union[int, str] = None,
    config: any=None,
) -> nn.Module:
    """
    Load a model from a wandb run ID.
    
    Parameters:
        run_id (str): A full wandb run id like "hazy-research/attention/159o6asi"
    """
    if config is not None and hasattr(config, "code_path"):
        sys.path.append(config.code_path)

    # 1: Get configuration from wandb
    config: Config = Config.from_wandb(run_id)
    path = config.checkpointer.dirpath

    # 2: Instantiate model
    model = config.model.instantiate()

    # 3: Load model
    # load the state dict, but remove the "model." prefix and all other keys from the
    # the PyTorch Lightning module that are not in the actual model
    ckpt = torch.load(os.path.join(path, "last.ckpt"), map_location=torch.device(device))

    print("Checkpoint Path: " + path)

    model.load_state_dict({
        k[len("model."):]: v 
        for k, v in ckpt["state_dict"].items() 
        if k.startswith("model.")
    })
    model.to(device=device)

    # 4: load tokenizer if it's available
    if hasattr(config.datamodule, "tokenizer_name"):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(config.datamodule.tokenizer_name)
    else:
        tokenizer = None


    return model, tokenizer
