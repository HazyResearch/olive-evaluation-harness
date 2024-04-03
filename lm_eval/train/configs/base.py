from train.config import *
from train.datamodules.language_modeling_hf import HFLanguageModelDataModuleConfig

config = Config(
    task="train.tasks.seq.SequenceLMModel",
    trainer=TrainerConfig(
        devices=1
    ),
    model=,
    datamodule=HFLanguageModelDataModuleConfig(
        dataset_name="wikitext",
        dataset_config_name="wikitext-103-v1",
        tokenizer_name="google/gemma-2b-it",
        cache_dir="/var/cr01_data/sabri/code/states/data/wikitext103/cache",
        batch_size=2
    ),
    optimizer=AdamOptimizerConfig(
        lr=1e-5
    ),
    optimizer_groups=OptimizerGroupsConfig(),
    scheduler=TimmCosineLRSchedulerConfig(
        warmup_lr_init = 1.0e-06,
        lr_min = 6e-08
    ),
    loss_fn=CrossEntropyLossConfig(),
    logger=WandbLoggerConfig(),
    checkpointer=ModelCheckpointConfig(),
    callbacks=[
        LearningRateMonitorConfig(),
        SpeedMonitorConfig(),
        LossScaleMonitorConfig(),
        GpuAffinityConfig(),
        RichModelSummaryConfig()
    ]
)