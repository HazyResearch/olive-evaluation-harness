"""
Launch a sweep of experiments (optionally in parallel using Ray).
"""
import importlib
from datetime import datetime
import os
import importlib.util
from typing import List

import click
from tqdm import tqdm

from train.config import Config, BaseConfig
from train.training import train


def execute_config(config: Config):
    try: 
        train(config=config)
    except Exception as e:
        return config, e
    return config, None


def _update_config(config: Config, updates: List[str]) -> Config:
    # perform nested updates in place
    for update in updates:
        arg_path, value = update.split("=")

        child, parent, relation = config, None, None
        for key in arg_path.split("."):
            next_node = getattr(child, key)
            if isinstance(next_node, BaseConfig):
                parent = child
                child = next_node
                relation = key
        if parent is None:
            child = child.model_validate({**child.to_dict(), key: value})
        else:
            setattr(parent, relation, child.model_validate({**child.to_dict(), key: value}))
    return config


@click.command()
@click.argument("python_file", type=click.Path(exists=True))
@click.option("-p", "--parallelize", is_flag=True)
@click.option("--gpus", default=None, type=str)
@click.option("--updates", "-u", type=str, multiple=True, help="Update config with these key=value pairs.")
def main(python_file, parallelize: bool, gpus: str, updates: List[str]):

    if gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus

    # Load the given Python file as a module
    spec = importlib.util.spec_from_file_location("config_module", python_file)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)

    # update configs with command line updates
    configs: List[Config] = config_module.configs
    launch_ids = []
    for config in configs:
        _update_config(config, updates)
        sweep_id = config.sweep_id
        config.launch_id = f"{sweep_id}-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
        launch_ids.append(config.launch_id)
    launch_ids = set(launch_ids)

    use_ray = parallelize and len(configs) > 0
    if use_ray:
        import ray
        # SE(03/02): ray was killing workers due to OOM, but it didn't seem to be necessary 
        os.environ["RAY_memory_monitor_refresh_ms"] = "0"
        ray.init(ignore_reinit_error=True, log_to_driver=False)

    print(f"Running sweeps {launch_ids} with {len(configs)} configs")

    # Run each script in parallel using Ray
    if not use_ray:
        for config in configs: 
            train(config)
    else:
        completed = 0
        failed = 0
        total = len(configs)
        print(f"Completed: {completed} ({completed / total:0.1%}) | Total: {total}")

        # we set the number of gpus required by each remote equal to the number of
        # gpus required by each config
        futures = [
            ray.remote(num_gpus=config.trainer.devices)(execute_config).remote(config) 
            for config in configs
        ]
        
        while futures:
            complete, futures = ray.wait(futures)
            for config, error in ray.get(complete):
                if error is not None:
                    failed += 1
                    config.print()
                    print(error)
                completed += 1
            print(f"Completed: {completed} ({completed / total:0.1%} -- {failed} failed) | Total: {total}")

        ray.shutdown()



if __name__ == "__main__":
    main()
