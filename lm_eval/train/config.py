from __future__ import annotations
from typing import Optional, List, Tuple, Union, Iterable, Dict, TYPE_CHECKING

from train.utils.utils import import_object, unflatten_dict
from pydantic import BaseModel, field_validator, Field

if TYPE_CHECKING:
    from transformers.configuration_utils import PretrainedConfig
    from pytorch_lightning import Trainer

class BaseConfig(BaseModel):

    def get(self, key, default=None):
        return getattr(self, key, default)

    def to_dict(self):
        return self._to_dict(self)
    
    def _to_dict(self, obj: any):
        if isinstance(obj, BaseConfig):
            data = {
                "_config_type": obj.__class__.__module__ + '.' + obj.__class__.__name__
            }
            for k, v in obj:
                data[k] = self._to_dict(v)
            return data
        elif isinstance(obj, list):
            return [self._to_dict(i) for i in obj]
        elif isinstance(obj, dict):
            return {k: self._to_dict(v) for k, v in obj.items()}
        else:
            return obj        
    
    @classmethod
    def from_dict(cls, data: Dict):
        if "_config_type" in data:
            cls = import_object(data["_config_type"])

        def _is_config(v):
            return isinstance(v, dict) and "_config_type" in v
        result = {}
        for k, v in data.items():
            if _is_config(v):
                result[k] = cls.from_dict(v)
            elif isinstance(v, list):
                result[k] = [cls.from_dict(i) if _is_config(i) else i for i in v]
            elif isinstance(v, dict):
                result[k] = {k: cls.from_dict(v) if _is_config(v) else v for k, v in v.items()}
            else:
                result[k] = v
        return cls(**result)
    
    @classmethod
    def from_wandb(cls, run_id: str):
        """
        Load a config from a wandb run ID.
    
        Parameters:
            run_id (str): A full wandb run id like "hazy-research/attention/159o6asi"
        """
        import wandb

        # 1: Get configuration from wandb
        api = wandb.Api()
        run = api.run(run_id)
        config = unflatten_dict(run.config)
        
        if "callbacks" in config and isinstance(config["callbacks"][0], str):
            # SE (04/01): this is a hack to deal with a bug in an old version of 
            # to_dict that didn't work with lists
            # eventually this can be removed when all configs are updated
            config["callbacks"] = []

        return cls.from_dict(config)

    def print(self):
        try:
            import rich
            rich.print(self)
        except ImportError:
            print(self)


class Config(BaseConfig):
    task: str = "train.tasks.seq.SequenceLMModel"
    trainer: TrainerConfig
    datamodule: DataModuleConfig
    model: BaseConfig
    optimizer: AdamOptimizerConfig
    optimizer_groups: OptimizerGroupsConfig
    loss_fn: ModuleConfig
    callbacks: List[CallbackConfig]
    checkpointer: ModelCheckpointConfig
    logger: WandbLoggerConfig

    # learning rate scheduler
    scheduler: LRSchedulerConfig
    schedular_interval: Optional[str] = "step"
    scheduler_monitor: Optional[str] = "val/loss"

    global_batch_size: int = 4

    seed: int = 1111
    test_only: bool = False
    test_after_training: bool = True
    optimized_metric: Optional[str] = None

    # resuming training and loading weights either from self or from a path
    # if both are set, self takes precedence if it exists
    resume_from_self: bool = True
    resume_from_path: Optional[str] = None  # include the full path to the checkpoint including the filename (e.g. /path/to/last.ckpt)

    # load a pretrained model and/or optimizer prior to training
    # note that both the pretrained paths will be overwritten by the resume_from_path
    # or resume_from_self options if they are set 
    pretrained_model_path: Optional[str] = None 
    pretrained_optimizer_path: Optional[str] = None

    # identifiers - useful for grouping, filtering, and labeling in wandb
    name: Optional[str] = None
    launch_id: Optional[str] = "default"
    sweep_id: Optional[str] = "default"


class ObjectConfig(BaseConfig):
    target: str
    kwargs: Optional[Dict] = Field(default_factory=dict)
    _pass_as_config: bool = False

    def instantiate(self, *args, **kwargs):
        print("Instantiating", self.target)
        cls = import_object(self.target)
        if self._pass_as_config:
            return cls(self, *args, **self.kwargs, **kwargs)

        # kwargs will overwrite the fields in the config
        return cls(
            *args,
            **self.kwargs,
            **kwargs,
            **self.model_dump(exclude={"target", "kwargs"} | set(kwargs.keys())),
        )


class TrainerConfig(ObjectConfig):
    target: str = "pytorch_lightning.Trainer"

    # There are several different counts that are logged to wandb and progress bars. 
    # Important to keep in mind the differences when setting interval based
    # parameters in this config
    #  - `batch_idx`: this is how many batches have been seen PER DEVICE, which is independent of 
    #     how many gradient steps there have been, depending on what `grad_accumulates`
    #     is set to. The tqdm progress bar in the terminal shows this number.
    #  - `trainer.global_step`: this is  accessible in PyTorch lightning by `trainer.global_step`
    #     This the number of optimizer steps taken: https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#global-step 
    #     This is equal to `batch_idx  // grad_accumulates`
    #  - `Step`: - this seems to be incremented every time wandb log is called. https://docs.wandb.ai/guides/integrations/lightning

    accelerator: str = "gpu"
    min_epochs: int = 1
    max_epochs: int = 1000
    devices: int = 8
    num_nodes: int = 1
    accumulate_grad_batches: int = 2
    max_steps: int = 800000
    log_every_n_steps: int = 50  # this is in terms of optimizer steps (trainer.global_step)
    val_check_interval: Union[float, int] = 4000  # this in terms of the number of batches seen per device, `batch_idx`
    check_val_every_n_epoch: Optional[int] = None
    precision: str = "bf16"
    gradient_clip_val: float = 1.0
    strategy: Optional[str] = "ddp"

    enable_checkpointing: bool = True
    overfit_batches: Union[int, float] = 0.0
    track_grad_norm: Union[int, float, str] = -1
    min_steps: Optional[int] = None
    max_time: Optional[Union[str, Dict[str, int]]] = None
    limit_train_batches: Optional[Union[int, float]] = None
    limit_val_batches: Optional[Union[int, float]] = None
    limit_test_batches: Optional[Union[int, float]] = None
    limit_predict_batches: Optional[Union[int, float]] = None
    sync_batchnorm: bool = False
    num_sanity_val_steps: int = 2
    deterministic: Optional[bool] = False
    reload_dataloaders_every_n_epochs: int = 0
    auto_lr_find: Union[bool, str] = False
    replace_sampler_ddp: bool = True
    auto_scale_batch_size: Union[str, bool] = False
    multiple_trainloader_mode: str = "max_size_cycle"
    inference_mode: bool = True

class DataModuleConfig(ObjectConfig):

    # this is the per-device batch size
    # together with the global_batch_size and number of devices, it determines
    # the number of grad accumulation steps
    batch_size: int
    tokenizer_name: str = "google/gemma-2b-it"

    @field_validator("target", check_fields=False)
    @classmethod
    def checktarget(cls, v: str):
        from pytorch_lightning import LightningDataModule
        obj = import_object(v)
        assert issubclass(obj, LightningDataModule)


class TorchModuleConfig(ObjectConfig):
    target: str = "torch.nn.Module"

    @field_validator("target", check_fields=False)
    @classmethod
    def checktarget(cls, v: str):
        from torch.nn import Module
        obj = import_object(v)
        assert issubclass(obj, Module)
        return v

class HFModelConfig(BaseConfig):
    pretrained_model_name_or_path: Optional[str] = None

    # regex to match the model parameter names that should be trained at inference
    # if None, all parametersare trained
    trainable_params: str = None
    load_kwargs: Optional[Dict] = Field(default_factory=dict)

    def instantiate(self):
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            self.pretrained_model_name_or_path,
            **self.load_kwargs
        )
        return model


class AdamOptimizerConfig(ObjectConfig):
    target: str = "apex.optimizers.FusedAdam"
    adam_w_mode: bool = True
    lr: float = 0.0006
    weight_decay: float = 0.1
    betas: Tuple[float, float] = (0.9, 0.95)

class OptimizerGroupsConfig(BaseConfig):
    bias_weight_decay: bool = False
    normalization_weight_decay: bool = False

class LRSchedulerConfig(ObjectConfig):
    @field_validator("target", check_fields=False)
    @classmethod
    def checktarget(cls, v: str):
        from torch.optim.lr_scheduler import _LRScheduler
        obj = import_object(v)
        assert issubclass(obj, _LRScheduler)
        return v

class TimmCosineLRSchedulerConfig(LRSchedulerConfig):
    target: str = "train.optim.timm_lr_scheduler.TimmCosineLRScheduler"
    t_in_epochs: bool = False

    # the  
    t_initial: int = 600000
    
    warmup_prefix: bool = True
    warmup_lr_init: float = 1.0e-06
    # the number of gradient steps (or epochs if t_in_epochs) to go from the 
    # warmup_lr_init to the lr
    warmup_t: float = 8000.0
    
    t_initial: int = 600000
    lr_min: float = 5.9999999999999995e-05

class ModuleConfig(ObjectConfig):
    @field_validator("target", check_fields=False)
    @classmethod
    def checktarget(cls, v: str):
        from torch.nn import Module
        obj = import_object(v)
        assert issubclass(obj, Module)
        return v

class CrossEntropyLossConfig(ModuleConfig):
    target: str = "flash_attn.losses.cross_entropy.CrossEntropyLoss"
    inplace_backward: bool = True

class WandbLoggerConfig(ObjectConfig):
    target: str = "pytorch_lightning.loggers.wandb.WandbLogger"
    project: str = "olive"
    name: Optional[str] = None
    save_dir: str = "."
    mode: str = "online"
    id: Optional[str] = None
    log_model: bool = False
    prefix: str = ""
    job_type: str = "train"
    group: str = ""
    tags: List[str] = []

class CallbackConfig(ObjectConfig):
    @field_validator("target", check_fields=False)
    @classmethod
    def checktarget(cls, v: str):
        from pytorch_lightning import Callback
        obj = import_object(v)
        assert issubclass(obj, Callback)
        return v

class ModelCheckpointConfig(CallbackConfig):
    target: str = "pytorch_lightning.callbacks.ModelCheckpoint"
    monitor: str = "val/loss"
    mode: str = "min"
    save_top_k: int = 0
    save_last: bool = True
    verbose: bool = False
    dirpath: str = "/var/cr01_data/sabri_data/checkpoints/"
    filename: str = "step_{step}"
    auto_insert_metric_name: bool = False
    every_n_train_steps: Optional[int] = 10_000
    every_n_epochs: Optional[int] = None

class LearningRateMonitorConfig(CallbackConfig):
    target: str = "pytorch_lightning.callbacks.LearningRateMonitor"
    logging_interval: str = "step"

class SpeedMonitorConfig(CallbackConfig):
    target: str = "train.callbacks.speed_monitor.SpeedMonitor"
    intra_step_time: bool = True
    inter_step_time: bool = True
    epoch_time: bool = True

class LossScaleMonitorConfig(CallbackConfig):
    target: str = "train.callbacks.loss_scale_monitor.LossScaleMonitor"

class ParamsLogConfig(CallbackConfig):
    target: str = "train.callbacks.params_log.ParamsLog"
    total_params_log: bool = True
    trainable_params_log: bool = True
    non_trainable_params_log: bool = True

class GpuAffinityConfig(CallbackConfig):
    target: str = "train.callbacks.gpu_affinity.GpuAffinity"

class NormMonitorConfig(CallbackConfig):
    target: str = "train.callbacks.norm_monitor.NormMonitor"

class RichModelSummaryConfig(CallbackConfig):
    target: str = "pytorch_lightning.callbacks.RichModelSummary"
