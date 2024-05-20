
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

@register_model("olive")
class OliveLMWrapper(HFLM):
    def __init__(
            self, 
            checkpoint_name: str,
            config: any=None,
            max_length: int = 2048,
            device: str = "cuda",
            **kwargs
        ) -> None:

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
        
        # downstream lm eval harness code expects to a property for the device
        model.device = device

        # 4: load tokenizer if it's available
        if hasattr(config.datamodule, "tokenizer_name"):
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(config.datamodule.tokenizer_name)
        else:
            tokenizer = None
        
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

    def _model_generate(self, context, max_length, stop, **generation_kwargs):
        for key in ("do_sample", "attention_mask"):
            if key in generation_kwargs:
                generation_kwargs.pop(key)

        # olive's custom GenerationMixin currently does not support
        # passing stopping criteria.
        # for the time being, we simply generate to max length,
        # then truncate (equivalent result)
        # -- this should be revisited to speed up generation
        # stopping_criteria = stop_sequences_criteria(
        #     self.tokenizer, stop, 1, context.shape[0]
        # )

        #breakpoint()

        """
        current_context = context[10].unsqueeze(0)
        decoded_prompt = self.tokenizer.decode(current_context)
        print(decoded_prompt)
        output = self.model.generate(input_ids=current_context,max_length=max_length,**generation_kwargs,)
        decoded_output = self.tokenizer.decode(output[0])
        print(decoded_output)



        prompt = "Question: " + requests[0].doc['question'] + " Answer:"
        #prompt = requests[i].doc['question']

        input = self.tokenizer.encode(prompt, return_tensors="pt").to("cuda")
        output = self.model.generate(input, max_length=32)
            
        decoded_output = self.tokenizer.decode(output[0])
        print(decoded_output)
        """

        return self.model.generate(
            input_ids=context,
            max_length=max_length,
            # stopping_criteria=stopping_criteria,
            # pad_token_id=self.tokenizer.pad_token_id,
            # use_cache=True,
            **generation_kwargs,
        )
    
    ################################################

    """def generate_until(self, requests):
        
        res = []

        #breakpoint()

        print("Performing generate_until!")
        for i in tqdm(range(len(requests))):
            
            prompt = "Question: " + requests[0].doc['question'] + " Answer:"
            #prompt = requests[i].doc['question']

            input = self.tokenizer.encode(prompt, return_tensors="pt").to("cuda")
            output = self.model.generate(input, max_length=32)
            
            decoded_output = self.tokenizer.decode(output[0])
            #print(decoded_output)
            assert decoded_output.strip() != ""

            res.append(decoded_output)

        ########################

        return res"""
    


@register_model("olive-recipes")
class OliveRecipesLMWrapper(HFLM):
    def __init__(
        self, 
        checkpoint_name: str,
        config: any=None,
        max_length: int = 2048,
        device: str = "cuda",
        **kwargs
    ) -> None:
        from llama_recipes.configs.training import TrainConfig
        from llama_recipes.model_checkpointing.checkpoint_handler import get_save_dir, load_model_checkpoint

        # 1: Get configuration from wandb
        #config: Config = Config.from_wandb(run_id)
        
        config: TrainConfig = TrainConfig.from_wandb(checkpoint_name)

        # 2: Instantiate model
        model = config.model.instantiate(train_config=config)

        load_model_checkpoint(model, rank=0, cfg=config)
        model.to(device=device)

        # 4: load tokenizer if it's available
        if config.tokenizer_name is not None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(config.datamodule.tokenizer_name)
        else:
            tokenizer = None
        
        super().__init__(
            pretrained=model,
            # set appropriate defaults for tokenizer, max length, etc
            #backend=kwargs.get("backend", "causal"),
            max_length=max_length,
            tokenizer=tokenizer,
            device=device,
            **kwargs,
        )
