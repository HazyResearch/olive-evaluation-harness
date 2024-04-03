from train.config import *
from train.datamodules.facts import FactsDataModuleConfig
from models.gemmix.configuration_gemmix import GemmixModelConfig, GemmixConfig


configs = []
for max_facts in [10_000]:
    for lr in [
        # 1e-4, 
        5e-5, 
        # 1e-5
    ]:
        for layer_idx in [16]:
            for num_experts in [1, 2, 4, 6, 8]:
                if num_experts == 1:
                    model=HFModelConfig(
                        pretrained_model_name_or_path="google/gemma-2b-it",
                        trainable_params=rf"model.layers.{layer_idx}.mlp.*",
                    )
                else:
                    model=GemmixModelConfig(
                        num_local_experts=num_experts, 
                        num_experts_per_tok=min(4, num_experts),
                        moe_layers=[layer_idx],
                        pretrained_model_name_or_path="google/gemma-2b-it",
                        trainable_params=rf"model.layers.{layer_idx}.mlp.*"
                    )
                    
                config = Config(
                    task="train.tasks.facts.FactsModel",
                    name=f"03-06-max_facts_{max_facts}_lidx{layer_idx}-lr_{lr}-num_exp_{num_experts}",
                    trainer=TrainerConfig(
                        devices=1,
                        val_check_interval=1.0, # checks after every epoch
                        check_val_every_n_epoch=1,
                        max_epochs=32
                    ),
                    global_batch_size=4,
                    model=model,
                    datamodule=FactsDataModuleConfig(
                        dataset_name="sabrieyuboglu/soccer-facts",
                        # dataset_config_name="wikitext-103-v1",
                        tokenizer_name="google/gemma-2b-it",
                        cache_dir="/var/cr01_data/sabri/code/olive/data/soccer_facts/cache",
                        batch_size=4,
                        use_shmem=True,
                        max_facts=max_facts
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

