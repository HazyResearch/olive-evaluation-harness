import os
from typing import List, Optional
from datetime import datetime
import json

import wandb
import click
from tqdm import tqdm

from train.config import Config


MAX_WORKERS_PER_GPU = 1


def execute_config(
    model: str,
    #run_id: str,
    task: str,
    batch_size: int,
    limit: int,
    output_dir: str,
    num_fewshot: int,
):
    # Save the original standard output
    import subprocess

    #output_dir = os.path.join(output_dir, model, run_id, task)
    output_dir = os.path.join(output_dir, model, task)

    args = [
        "lm_eval",
        "--model", "olive", #"based_lm"
        "--model_args", f"checkpoint_name={model}",
        #"--model_args", f"checkpoint_name={run_id}",
        "--tasks", task,
        "--device", "cuda:0",
        "--batch_size", str(batch_size),
        "--log_samples",
        "--output_path", output_dir,
        "--num_fewshot", str(num_fewshot)
    ]

    if limit is not None:
        args.extend(["--limit", str(limit)])
    try:
        subprocess.run(args)

        # upload results to wandb
        results = json.load(open(os.path.join(output_dir, "results.json")))
        train_config = Config.from_wandb(model)
        wandb.init(
            project="olive-eval",
            name=f"{task}-{train_config.name}",
            config={
                "train": train_config.to_dict(),
                "task": results["configs"][task],
                **results["config"],
                "git_hash": results["git_hash"],
                "run_id": model,
            }
        )
        wandb.log({
            f"{task}/{k}": v
            for k,v in results["results"][task].items()
        })
        wandb.finish()
        return args, None
    except Exception as e:
        return args, e



@click.command()
@click.option("-m", "--model", type=str, multiple=True)
#@click.option("-m", "--run_id", type=str, multiple=True)
@click.option("-t", "--task", type=str, multiple=True)
@click.option("-p", "--parallelize", is_flag=True)
@click.option("--gpus", default=None, type=str)
@click.option("--batch-size", default=8, type=int)
@click.option("--limit", default=None, type=int)
@click.option("--num_fewshot", default=0, type=int)
def main(
    model: List[str],
    #run_id: List[str],
    task: List[str], 
    batch_size: int,
    limit: Optional[int],
    parallelize: bool, 
    gpus: str,
    num_fewshot: int = 0,
):

    if gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus

    # Load the given Python file as a module
    configs = [
        {"model": m, "task": t} for m in model for t in task
        #{"model": m, "run_id": id, "task": t} for m in model for t in task for id in run_id
    ]

    use_ray = parallelize and len(configs) > 0
    if use_ray:
        import ray
        # ray was killing workers due to OOM, but it didn't seem to be necessary 
        os.environ["RAY_memory_monitor_refresh_ms"] = "0"
        ray.init(ignore_reinit_error=True, log_to_driver=True)

    print(f"Running sweep with {len(configs)} configs")

    output_dir = f"output/{datetime.now().strftime('%y-%m-%d_%H-%M')}"

    # Run each script in parallel using Ray
    if not use_ray:
        for config in configs: 
            execute_config(
                **config,
                batch_size=batch_size,
                limit=limit,
                output_dir=output_dir,
                num_fewshot=num_fewshot,
            )
    else:
        completed = 0
        failed = 0
        total = len(configs)
        print(f"Completed: {completed} ({completed / total:0.1%}) | Total: {total}")

        remote = ray.remote(num_gpus=(1 // MAX_WORKERS_PER_GPU))(execute_config)
        futures = [remote.remote(**config, batch_size=batch_size, limit=limit, output_dir=output_dir, num_fewshot=num_fewshot) for config in configs]
        
        while futures:
            complete, futures = ray.wait(futures)
            for config, error in ray.get(complete):
                if error is not None:
                    failed += 1
                    print(config)
                    print(error)
                completed += 1
            print(f"Completed: {completed} ({completed / total:0.1%} -- {failed} failed) | Total: {total}")

        ray.shutdown()



if __name__ == "__main__":
    main()
