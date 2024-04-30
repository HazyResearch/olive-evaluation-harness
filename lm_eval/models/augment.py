

import re
from transformers import AutoTokenizer
import torch

from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM
#from api.registry import register_model
#from models.huggingface import HFLM

import os
import sys
import torch 
import wandb
from tqdm import tqdm

from train.config import Config
from train.modeling_llama_flash import LlamaForCausalContextLM

import torch
from transformers import LlamaTokenizer
#from modeling_llama_flash import LlamaForCausalContextLM

@register_model("augment")
class AugmentLMWrapper(HFLM):
    def __init__(
            self, 
            checkpoint_name: str,
            model_name: str,
            config: any=None,
            max_length: int = 2048,
            device: str = "cuda",
            **kwargs
        ) -> None:

        ########################################################

        if checkpoint_name is not None:

            if config is not None and hasattr(config, "code_path"):
                sys.path.append(config.code_path)

            # 1: Get configuration from wandb
            #config: Config = Config.from_wandb(run_id)
            
            config: Config = Config.from_wandb(checkpoint_name)
            path = config.checkpointer.dirpath

            # 2: Instantiate model
            model = config.model.instantiate()

            # 3: Load model
            # load the state dict, but remove the "model." prefix and all other keys from the
            # the PyTorch Lightning module that are not in the actual model
            try:
                ckpt = torch.load(os.path.join(path, "last.ckpt"), map_location=torch.device(device))
            except: # FileNotFoundError:
                try:
                    run_id = checkpoint_name.split("/")[-1]
                    path = os.path.join(config.checkpointer.dirpath, config.sweep_id, config.name, run_id)
                    ckpt = torch.load(os.path.join(path, "last.ckpt"), map_location=torch.device(device))
                except:
                    ckpt = torch.load(config.checkpointer.dirpath + config.name + "/" + run_id + "/" + "last.ckpt", map_location=torch.device(device))
                    

            print("Checkpoint Path: " + path)

            model.load_state_dict({
                k[len("model."):]: v 
                for k, v in ckpt["state_dict"].items() 
                if k.startswith("model.")
            })
            model.to(device=device)

        ########################################################

        else:

            model = LlamaForCausalContextLM.from_pretrained(
                model_name,
                use_flash_attention_2="flash_attention_2", 
                torch_dtype=torch.bfloat16,
                device_map="auto",
            ).eval()

        ########################################################
        
        # downstream lm eval harness code expects to a property for the device
        model.device = device

        # 4: load tokenizer if it's available
        if hasattr(config.datamodule, "tokenizer_name"):
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(config.datamodule.tokenizer_name)
        else:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        super().__init__(
            pretrained=model,
            # set appropriate defaults for tokenizer, max length, etc
            #backend=kwargs.get("backend", "causal"),
            max_length=max_length,
            tokenizer=tokenizer,
            device=device,
            **kwargs,
        )

    ################################################

    def _model_generate(self, context, max_length, stop, encoder_contexts, **generation_kwargs):
        
        for key in ("do_sample", "attention_mask"):
            if key in generation_kwargs:
                generation_kwargs.pop(key)

        return self.model.generate(
            input_ids=context.input_ids, 
            attention_mask=context.attention_mask, 
            encoder_input_ids=encoder_contexts.input_ids.unsqueeze(0),
            encoder_attention_mask=encoder_contexts.attention_mask, 
            max_new_tokens=200,
            #sample=True,
            top_p=0.95,
            output_hidden_states=True
            )
