import os
import numpy as np

from train.config import *
from train.datamodules.language_modeling_hf import HFLanguageModelDataModuleConfig
from train.datamodules.language_modeling_neox import NeoxLMDataModuleConfig
from olive.models.flash_gpt.config import FlashGPTModelConfig


configs = []

# setup to match 
# https://wandb.ai/hazy-research/based/runs/02-21-attn-360m-redo1/overview?nw=nwuserseyuboglu

max_steps = 20_000
global_batch_size = 256
lr=8e-4
d_model = 1024
multiple_of = 128
d_hidden = int(d_model * 8 / 3)
d_hidden = (d_hidden + multiple_of - 1) // multiple_of * multiple_of


model=FlashGPTModelConfig(
    n_embd=d_model,
    n_layer=12,
    n_head=16,
    n_positions=0,
    vocab_size=50257,
    activation_function="swiglu",
    max_position_embeddings=0,
    custom_mlp=TorchModuleConfig(
        target="olive.layers.scattermoe.mlp.ScatterMoEBlock",
        kwargs=dict(
            d_model=d_model,
            d_hidden=d_hidden,
            num_experts=8,
            top_k=2
        )
    )
)
config = Config(
    name=f"simple_moe",
    sweep_id=os.path.splitext(os.path.basename(__file__))[0],
    task="train.tasks.language_model.LanguageModel",
    trainer=TrainerConfig(
        devices=1,
        check_val_every_n_epoch=None,
        val_check_interval=2000,
        max_steps=max_steps,
        log_every_n_steps=50,
        limit_test_batches=50, 
        limit_val_batchs=50,
    ),
    resume_from_self=False,
    global_batch_size=global_batch_size,
    model=model,
    datamodule=NeoxLMDataModuleConfig(
        max_length=2048,
        batch_size=16,
        batch_size_eval=32,
        max_steps=20_000,
        pin_memory=True,  
        tokenizer_name=f"gpt2",
        num_train_samples=max_steps * global_batch_size,
        num_predict_batches_eval=100
    ),
    
    optimizer=AdamOptimizerConfig(
        lr=lr
    ),
    optimizer_groups=OptimizerGroupsConfig(),
    scheduler=TimmCosineLRSchedulerConfig(
        warmup_lr_init=1e-6,
        warmup_t= 200.0,
        lr_min=8e-5,
        t_initial=19800
    ),
    loss_fn=CrossEntropyLossConfig(),
    logger=WandbLoggerConfig(),
    checkpointer=ModelCheckpointConfig(
        # don't save checkpoints, to save space
        save_last=True,
        save_top_k=0,
        every_n_train_steps=10_000,
        every_n_epochs=None
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
