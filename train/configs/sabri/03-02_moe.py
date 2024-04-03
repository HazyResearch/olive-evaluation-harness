from train.config import *
from train.datamodules.language_modeling_hf import HFLanguageModelDataModuleConfig
from models.gemmix.configuration_gemmix import GemmixModelConfig, GemmixConfig

configs = []
for num_experts in [2, 4, 6, 8]:
    for layer_idx in [15, 16, 14]:

        config = Config(
            task="train.tasks.seq.SequenceLMModel",
            name=f"03-03_num_experts_{num_experts}_{layer_idx}",
            trainer=TrainerConfig(
                devices=1,
                val_check_interval=500,
                max_epochs=2
            ),
            global_batch_size=4,
            model=GemmixModelConfig(
                num_local_experts=num_experts, 
                num_experts_per_tok=min(2, num_experts),
                moe_layers=[15],
                pretrained_model_name_or_path="google/gemma-2b-it",
                trainable_params=r"model.layers.15.mlp.*"
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
            checkpointer=ModelCheckpointConfig(
                # don't save checkpoints, to save space
                save_last=False,
                save_top_k=0
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

