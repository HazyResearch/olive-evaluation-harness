import os

from train.config import *
from train.datamodules.facts import FactsDataModuleConfig
from models.gpt_neox.configuration_gpt_neox import NeoxModelConfig

configs = []
for max_facts in [
    #   1_000, 
    #   2_000, 
    #   4_000, 
    # 8_000, 
    #  16_000, 
    #   32_000, 
    # 64_000, 
    # 128_000,
    # 256_000,
    512_000,
    # 64_000
]:
    for model_name in [
        # "pythia-70m",
        # "pythia-160m",
        "pythia-410m",
        # "pythia-1b",
        # "pythia-1.4b",
    ]:
        for moe_layer in [8]:
            for num_experts in [1, 2, 4, 6, 8, 10, 12, 16]:
                    for lr in [8e-5]:
                        model=NeoxModelConfig(
                            pretrained_model_name_or_path=f"EleutherAI/{model_name}",
                            # trainable_params=rf"gpt_neox.layers.{layer_idx}.mlp.*",
                            num_local_experts=num_experts,
                            num_experts_per_tok=1,
                            moe_layers=[moe_layer],
                            trainable_params=rf"gpt_neox.layers.({moe_layer}).mlp.*",
                        )

                        config = Config(
                            name=f"{model_name}_experts_{num_experts}_topk_{1}",
                            sweep_id=os.path.splitext(os.path.basename(__file__))[0],
                            task="train.tasks.facts.FactsModel",
                            trainer=TrainerConfig(
                                devices=1,
                                val_check_interval=1.0, # checks after every epoch
                                check_val_every_n_epoch=1,
                                max_epochs=128
                            ),
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
                                force_cache=True
                            ),
                            optimizer=AdamOptimizerConfig(lr=lr),
                            optimizer_groups=OptimizerGroupsConfig(),
                            scheduler=TimmCosineLRSchedulerConfig(
                                warmup_lr_init=lr,
                                lr_min=lr,  # no change in lr
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

