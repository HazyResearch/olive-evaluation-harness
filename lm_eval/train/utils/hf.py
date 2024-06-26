""" Source: https://github.com/state-spaces/mamba/blob/main/mamba_ssm/utils/hf.py """
import os
import json
import torch
from transformers.utils import WEIGHTS_NAME, CONFIG_NAME
from transformers.utils.hub import cached_file



def upload_model_hf(
    wandb_path: str="hazy-research/based/02-20-based-360m",
    hf_path: str=f"HazyResearch/based-360m",
):
    """
    Load a config from a wandb run ID.
    
    Parameters:
        run_id (str): A full wandb run id like "hazy-research/attention/159o6asi"
    """
    import wandb
    import hydra

    # 1: Get configuration from wandb
    api = wandb.Api()
    run = api.run(wandb_path)
    config = _unflatten_dict(run.config)
    path = config["callbacks"]["model_checkpoint"]["dirpath"]
    model_config = config["model"].pop("config")

    # make subdirectory in the model path where we will create a directory for upload 
    # to HF 
    local_hf_path = os.path.join(path, "hf")
    os.makedirs(local_hf_path, exist_ok=True)

    # 2: Create model
    model_config = hydra.utils.instantiate(
        model_config, _recursive_=False, _convert_="object"
    )
    json.dump(model_config.to_dict(), open(os.path.join(local_hf_path, CONFIG_NAME), "w"))

    # 3: Load model
    # load the state dict, but remove the "model." prefix and all other keys from the
    # the PyTorch Lightning module that are not in the actual model
    ckpt = torch.load(os.path.join(path, "last.ckpt"))
    

    state_dict = {
        k[len("model."):]: v 
        for k, v in ckpt["state_dict"].items() 
        if k.startswith("model.")
    }
    torch.save(state_dict, os.path.join(local_hf_path, WEIGHTS_NAME))

    # 4: Save model to HF
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(name=hf_path, repo_type="model", exist_ok=False)
    api.upload_folder(
        folder_path=local_hf_path, repo_id=hf_path, repo_type="model",
    )


    return config



def _unflatten_dict(d: dict) -> dict:
    """ 
    Takes a flat dictionary with '/' separated keys, and returns it as a nested dictionary.
    
    Parameters:
    d (dict): The flat dictionary to be unflattened.
    
    Returns:
    dict: The unflattened, nested dictionary.
    """
    result = {}

    for key, value in d.items():
        parts = key.split('/')
        d = result
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        d[parts[-1]] = value

    return result

if __name__ == "__main__":
    upload_model_hf()
