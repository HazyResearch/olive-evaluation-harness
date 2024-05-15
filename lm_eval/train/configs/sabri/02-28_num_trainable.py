from train.config import *
from train.datamodules.language_modeling_hf import HFLanguageModelDataModuleConfig

configs = []
for num_trainable in [1, 2, 3, 4]:
    if num_trainable == 1:
        trainable_params = r"model.layers.15.mlp.*"
    elif num_trainable == 2:
        trainable_params = r"model.layers.(15|16).mlp.*"
    elif num_trainable == 3:
        trainable_params = r"model.layers.(15|16|17).mlp.*"
    elif num_trainable == 4:
        trainable_params = r"model.layers.(15|16|17|18).mlp.*"
     

    config = Config(
        task="train.tasks.seq.SequenceLMModel",
        name=f"02-27_num_trainable_{num_trainable}",
        trainer=TrainerConfig(
            devices=1,
            val_check_interval=500
        ),
        global_batch_size=4,
        model=HFModelConfig(
            pretrained_model_name_or_path="google/gemma-2b-it",
            trainable_params=trainable_params
        ),
        datamodule=HFLanguageModelDataModuleConfig(
            dataset_name="wikitext",
            dataset_config_name="wikitext-103-v1",
            tokenizer_name="google/gemma-2b-it",
            cache_dir="/var/cr01_data/sabri/code/states/data/wikitext103/cache",
            batch_size=4
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
    configs.append(config)

