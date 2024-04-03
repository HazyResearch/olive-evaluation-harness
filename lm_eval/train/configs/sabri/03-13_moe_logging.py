import numpy as np

from train.config import *
from train.datamodules.facts import FactsDataModuleConfig
from models.gpt_neox.configuration_gpt_neox import NeoxModelConfig

max_facts = 16_000
model_name = "pythia-410m"
shuffle_seed=456
# shuffle_seed=123

configs = []
for lr in [8e-5]:

    moe_layer = 14
    model=NeoxModelConfig(
        pretrained_model_name_or_path=f"EleutherAI/{model_name}",
        num_local_experts=8,
        num_experts_per_tok=2,
        moe_layers=[moe_layer],
        trainable_params=rf"gpt_neox.layers.({moe_layer}).mlp.*",
        log_routing=True
    )

    config = Config(
        name=f"moe_logging",
        sweep_id="moe_logging",
        task="train.tasks.facts.FactsModel",
        trainer=TrainerConfig(
            devices=1,
            val_check_interval=1.0, # checks after every epoch
            check_val_every_n_epoch=1,
            max_epochs=32
        ),
        resume_from_self=False,
        global_batch_size=4,
        model=model,
        datamodule=FactsDataModuleConfig(
            dataset_name="sabrieyuboglu/soccer-facts",
            # dataset_config_name="wikitext-103-v1",
            tokenizer_name=f"EleutherAI/{model_name}",
            cache_dir="/var/cr01_data/sabri/code/olive/data/soccer_facts/cache",
            batch_size=4,
            use_shmem=True,
            max_facts=max_facts,
            force_cache=True,
            shuffle_seed=shuffle_seed
        ),
        optimizer=AdamOptimizerConfig(
            lr=lr
        ),
        optimizer_groups=OptimizerGroupsConfig(),
        scheduler=TimmCosineLRSchedulerConfig(
            warmup_lr_init=lr,
            lr_min=lr,
            t_initial=1000
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

