import os
import numpy as np

from train.config import *
from train.datamodules.language_modeling_hf import HFLanguageModelDataModuleConfig
from models.gpt_neox.configuration_gpt_neox import NeoxModelConfig

model_name = "pythia-410m"

configs = []
for lr in [1e-5, 1e-4, 1e-3]:
    for warmup_lr_init in [1e-7, 1e-6, 1e-5]:

        max_steps = 5000  # should be 20_000
        global_batch_size = 128

        # model=NeoxModelConfig(
        #     pretrained_model_name_or_path=f"EleutherAI/{model_name}",
        #     # num_local_experts=8,
        #     # num_experts_per_tok=2,
        #     # moe_layers=[],
        #     # trainable_params=rf"gpt_neox.layers.({moe_layer}).mlp.*",
        #     # log_routing=True
        # )
        model=HFModelConfig(
            pretrained_model_name_or_path=f"EleutherAI/{model_name}",
        )

        config = Config(
            name=f"moe_logging",
            sweep_id=os.path.splitext(os.path.basename(__file__))[0],
            task="train.tasks.language_model.LanguageModel",
            trainer=TrainerConfig(
                devices=1,
                check_val_every_n_epoch=None,
                val_check_interval=1000,
                max_steps=max_steps,
                log_every_n_steps=100,
                limit_test_batches=50
            ),
            resume_from_self=False,
            global_batch_size=global_batch_size,
            model=model,
            datamodule=HFLanguageModelDataModuleConfig(
                dataset_name="wikitext",
                dataset_config_name="wikitext-103-v1",
                tokenizer_name=f"EleutherAI/{model_name}",
                cache_dir="/var/cr01_data/sabri/code/states/data/wikitext103/cache",
                batch_size=2
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
