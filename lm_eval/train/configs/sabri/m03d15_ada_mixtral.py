import os
import numpy as np

from train.config import *
from train.datamodules.language_modeling_hf import HFLanguageModelDataModuleConfig
from train.datamodules.language_modeling_neox import NeoxLMDataModuleConfig
from models.ada_mixtral.configuration_ada_mixtral import AdaMixtralModelConfig

model_name = "pythia-410m"

configs = []
for lr in [1e-4, 1e-3]:
    for warmup_lr_init in [1e-6, 1e-5]:
        max_steps = 20_000
        global_batch_size = 256

        model=AdaMixtralModelConfig(
            hidden_size=512,
            intermediate_size=1024,
            num_hidden_layers=6,
            num_attention_heads=8,
            num_key_value_heads=8,
            max_position_embeddings=2048,
            vocab_size=50257,
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
                batch_size=32,
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
