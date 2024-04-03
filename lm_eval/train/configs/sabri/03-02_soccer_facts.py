from train.config import *
from train.datamodules.facts import FactsDataModuleConfig
from models.gemmix.configuration_gemmix import GemmixModelConfig, GemmixConfig

    

configs = []
for num_trainable in [1]:
    for lr in [5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6]:
        for batch_size in [8]:
            if num_trainable == 1:
                trainable_params = r"model.layers.15.mlp.*"
            elif num_trainable == 2:
                trainable_params = r"model.layers.(15|16).mlp.*"
            elif num_trainable == 3:
                trainable_params = r"model.layers.(15|16|17).mlp.*"
            elif num_trainable == 4:
                trainable_params = r"model.layers.(15|16|17|18).mlp.*"
            
            config = Config(
                task="train.tasks.facts.FactsModel",
                name=f"03-03_num_trainable_{num_trainable}-lr_{lr}-bs_{batch_size}",
                trainer=TrainerConfig(
                    devices=1,
                    val_check_interval=1.0, # checks after every epoch
                    check_val_every_n_epoch=1,
                    max_epochs=64
                ),
                global_batch_size=batch_size,
                model=HFModelConfig(
                    pretrained_model_name_or_path="google/gemma-2b-it",
                    trainable_params=trainable_params,
                ),
                datamodule=FactsDataModuleConfig(
                    dataset_name="sabrieyuboglu/soccer-facts",
                    # dataset_config_name="wikitext-103-v1",
                    tokenizer_name="google/gemma-2b-it",
                    cache_dir="/var/cr01_data/sabri/code/olive/data/soccer_facts/cache",
                    batch_size=batch_size,
                    use_shmem=True,
                    max_facts=10_000
                ),
                optimizer=AdamOptimizerConfig(
                    lr=lr
                ),
                optimizer_groups=OptimizerGroupsConfig(),
                scheduler=TimmCosineLRSchedulerConfig(
                    warmup_lr_init=lr,
                    lr_min = 6e-08,
                    t_initial=1000
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

