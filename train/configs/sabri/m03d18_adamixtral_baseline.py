import os
import numpy as np

from train.config import *
from train.datamodules.language_modeling_hf import HFLanguageModelDataModuleConfig
from train.datamodules.language_modeling_neox import NeoxLMDataModuleConfig
from models.ada_mixtral.configuration_ada_mixtral import AdaMixtralModelConfig


configs = []

# setup to match 
# https://wandb.ai/hazy-research/based/runs/02-21-attn-360m-redo1/overview?nw=nwuserseyuboglu

max_steps = 20_000
global_batch_size = 256
lr=8e-4
d_model = 1024

model=AdaMixtralModelConfig(
    hidden_size=d_model,
    intermediate_size=int(d_model * 8/3),
    num_hidden_layers=24,
    num_attention_heads=16,
    num_key_value_heads=16,
    max_position_embeddings=2048,
    vocab_size=50257,
    num_local_experts=1,
    num_experts_per_tok=1
)

config = Config(
    name=f"ada_mixtral_setup",
    sweep_id=os.path.splitext(os.path.basename(__file__))[0],
    task="train.tasks.language_model.LanguageModel",
    trainer=TrainerConfig(
        devices=1,
        check_val_every_n_epoch=None,
        val_check_interval=1000,
        max_steps=max_steps,
        log_every_n_steps=100,
        limit_test_batches=50, 
        limit_val_batchs=50,
    ),
    resume_from_self=False,
    global_batch_size=global_batch_size,
    model=model,
    
    # datamodule=HFLanguageModelDataModuleConfig(
    #     dataset_name="wikitext",
    #     dataset_config_name="wikitext-103-v1",
    #     tokenizer_name="mistralai/Mixtral-8x7B-v0.1",
    #     cache_dir="/var/cr01_data/sabri/code/states/data/wikitext103/cache",
    #     batch_size=32
    # ),
    datamodule=NeoxLMDataModuleConfig(
        max_length=2048,
        batch_size=16,
        max_steps=20_000,
        pin_memory=True,  
        tokenizer_name=f"gpt2",
        num_train_samples=max_steps * global_batch_size,
    ),
    
    optimizer=AdamOptimizerConfig(
        lr=lr
    ),
    optimizer_groups=OptimizerGroupsConfig(),
    scheduler=TimmCosineLRSchedulerConfig(
        warmup_lr_init=1e-6,
        warmup_t= 200.0,
        lr_min=lr * 0.1,  # from the Pythia paper
        t_initial=19800
    ),
    loss_fn=CrossEntropyLossConfig(),
    logger=WandbLoggerConfig(),
    checkpointer=ModelCheckpointConfig(
        # don't save checkpoints, to save space
        save_last=True,
        save_top_k=0,
        every_n_train_steps=None,
        every_n_epochs=1
    ),
    callbacks=[
        LearningRateMonitorConfig(),
        SpeedMonitorConfig(),
        LossScaleMonitorConfig(),
        GpuAffinityConfig(),
        RichModelSummaryConfig()
    ]
)
configs.append(config)

configs = configs[:1]
