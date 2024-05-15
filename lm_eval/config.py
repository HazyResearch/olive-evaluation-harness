import os
from typing import List, Optional
from datetime import datetime
import subprocess
import json
import argparse
import json
import logging
import os
import sys
from functools import partial
from typing import Optional, Union, List

import wandb
import click
from tqdm import tqdm
from pydantic import BaseModel, Field

from olive.config import RunConfig

from lm_eval import evaluator, utils
from lm_eval.evaluator import request_caching_arg_to_dict
from lm_eval.logging import EvaluationTracker, WandbLogger
from lm_eval.tasks import TaskManager
from lm_eval.utils import handle_non_serializable, make_table, simple_parse_args_string


class EvalConfig(RunConfig):
    model: str = Field("hf", description="Name of model e.g. `hf`")
    model_args: str = Field("", description="Comma separated string arguments for model, e.g. `pretrained=EleutherAI/pythia-160m,dtype=float32`")
    gen_kwargs: Optional[str] = Field(None, description="String arguments for model generation on greedy_until tasks, e.g. `temperature=0,top_k=0,top_p=0`.")

    tasks: Optional[str] = Field(None, description="To get full list of tasks, use the command lm-eval --tasks list")
    num_fewshot: Optional[int] = Field(None, description="Number of examples in few-shot context")
    limit: Optional[float] = Field(None, description="Limit the number of examples per task. If <1, limit is a percentage of the total number of examples.")


    batch_size: str = Field("1", description="Acceptable values are 'auto', 'auto:N' or N, where N is an integer. Default 1.")
    max_batch_size: Optional[int] = Field(None, description="Maximal batch size to try with --batch_size auto.")

    device: Optional[str] = Field(None, description="Device to use (e.g. cuda, cuda:0, cpu).")

    output_path: Optional[str] = Field(None, description="The path to the output file where the result metrics will be saved. If the path is a directory and log_samples is true, the results will be saved in the directory. Else the parent directory will be used.")
    
    use_cache: Optional[str] = Field(None, description="A path to a sqlite db file for caching model responses. `None` if not caching.")
    cache_requests: Optional[str] = Field(None, description="Speed up evaluation by caching the building of dataset requests. `None` if not caching.")
    check_integrity: bool = Field(False, description="Whether to run the relevant part of the test suite for the tasks.")
    
    wandb: bool = Field(False, description="Logs results to wandb")
    wandb_args: str = Field("", description="Comma separated string arguments passed to wandb.init, e.g. `project=lm-eval,job_type=eval")
    hf_hub_log_args: str = Field("", description="Comma separated string arguments passed to Hugging Face Hub's log function, e.g. `hub_results_org=EleutherAI,hub_repo_name=lm-eval-results`") 

    write_out: bool = Field(False, description="Prints the prompt for the first few documents.")
    log_samples: bool = Field(False, description="If True, write out all model outputs and documents for per-sample measurement and post-hoc analysis. Use with --output_path.")
    verbosity: str = Field("INFO", description="Controls the reported logging error level. Set to DEBUG when testing + adding new task configurations for comprehensive log output.")

    output_path: Optional[str] = Field(None, description="The path to the output file where the result metrics will be saved. If the path is a directory and log_samples is true, the results will be saved in the directory. Else the parent directory will be used.")
    show_config: bool = Field(False, description="If True, shows the the full config of all tasks at the end of the evaluation.")
    include_path: Optional[str] = Field(None, description="Additional path to include if there are external tasks to include.")
    
    verbosity: str = Field("INFO", description="Controls the reported logging error level. Set to DEBUG when testing + adding new task configurations for comprehensive log output.")
    predict_only: bool = Field(False, description="Use with --log_samples. Only model outputs will be saved and metrics will not be evaluated.")
    seed: Union[List[Optional[int]], int] = Field([0, 1234, 1234, 1234], description="Set seed for python's random, numpy, torch, and fewshot sampling.\nAccepts a comma-separated list of 4 values for python's random, numpy, torch, and fewshot sampling seeds, respectively, or a single integer to set the same seed for all three.\nThe values are either an integer or 'None' to not set the seed. Default is `(for backward compatibility).\nE.g. `--seed 0,None,8,52` sets `random.seed(0)`, `torch.manual_seed(8)`, and fewshot sampling seed to 52. Here numpy's seed is not set since the second value is `None`.\nE.g, `--seed 42` sets all four seeds to 42.")
    trust_remote_code: bool = Field(False, description="Sets trust_remote_code to True to execute code to create HF Datasets from the Hub")

    def run(self):
       evaluate(self)



MAX_WORKERS_PER_GPU = 1

def evaluate(args: EvalConfig) -> None:
    args.wandb_args = args.wandb_args + f",name={args.name}"


    if args.wandb_args:
        wandb_logger = WandbLogger(**simple_parse_args_string(args.wandb_args))

    eval_logger = utils.eval_logger
    eval_logger.setLevel(getattr(logging, f"{args.verbosity}"))
    eval_logger.info(f"Verbosity set to {args.verbosity}")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # update the evaluation tracker args with the output path and the HF token
    args.hf_hub_log_args = f"output_path={args.output_path},token={os.environ.get('HF_TOKEN')},{args.hf_hub_log_args}"
    evaluation_tracker_args = simple_parse_args_string(args.hf_hub_log_args)
    evaluation_tracker = EvaluationTracker(**evaluation_tracker_args)
    evaluation_tracker.general_config_tracker.log_experiment_args(
        model_source=args.model,
        model_args=args.model_args,
    )

    if args.predict_only:
        args.log_samples = True
    if (args.log_samples or args.predict_only) and not args.output_path:
        raise ValueError(
            "Specify --output_path if providing --log_samples or --predict_only"
        )

    if args.include_path is not None:
        eval_logger.info(f"Including path: {args.include_path}")
    task_manager = TaskManager(args.verbosity, include_path=args.include_path)

    if (
        "push_results_to_hub" in evaluation_tracker_args
        or "push_samples_to_hub" in evaluation_tracker_args
    ) and "hub_results_org" not in evaluation_tracker_args:
        raise ValueError(
            "If push_results_to_hub or push_samples_to_hub is set, results_org must be specified."
        )
    if "push_samples_to_hub" in evaluation_tracker_args and not args.log_samples:
        eval_logger.warning(
            "Pushing samples to the Hub requires --log_samples to be set. Samples will not be pushed to the Hub."
        )

    if args.limit:
        eval_logger.warning(
            " --limit SHOULD ONLY BE USED FOR TESTING."
            "REAL METRICS SHOULD NOT BE COMPUTED USING LIMIT."
        )

    if args.tasks is None:
        eval_logger.error("Need to specify task to evaluate.")
        sys.exit()
    elif args.tasks == "list":
        eval_logger.info(
            "Available Tasks:\n - {}".format("\n - ".join(task_manager.all_tasks))
        )
        sys.exit()
    else:
        if os.path.isdir(args.tasks):
            import glob

            task_names = []
            yaml_path = os.path.join(args.tasks, "*.yaml")
            for yaml_file in glob.glob(yaml_path):
                config = utils.load_yaml_config(yaml_file)
                task_names.append(config)
        else:
            task_list = args.tasks.split(",")
            task_names = task_manager.match_tasks(task_list)
            for task in [task for task in task_list if task not in task_names]:
                if os.path.isfile(task):
                    config = utils.load_yaml_config(task)
                    task_names.append(config)
            task_missing = [
                task for task in task_list if task not in task_names and "*" not in task
            ]  # we don't want errors if a wildcard ("*") task name was used

            if task_missing:
                missing = ", ".join(task_missing)
                eval_logger.error(
                    f"Tasks were not found: {missing}\n"
                    f"{utils.SPACING}Try `lm-eval --tasks list` for list of available tasks",
                )
                raise ValueError(
                    f"Tasks not found: {missing}. Try `lm-eval --tasks list` for list of available tasks, or '--verbosity DEBUG' to troubleshoot task registration issues."
                )

    # Respect user's value passed in via CLI, otherwise default to True and add to comma-separated model args
    if args.trust_remote_code:
        os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = str(args.trust_remote_code)
        args.model_args = (
            args.model_args
            + f",trust_remote_code={os.environ['HF_DATASETS_TRUST_REMOTE_CODE']}"
        )

    eval_logger.info(f"Selected Tasks: {task_names}")

    request_caching_args = request_caching_arg_to_dict(
        cache_requests=args.cache_requests
    )

    results = evaluator.simple_evaluate(
        model=args.model,
        model_args=args.model_args,
        tasks=task_names,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        max_batch_size=args.max_batch_size,
        device=args.device,
        use_cache=args.use_cache,
        limit=args.limit,
        check_integrity=args.check_integrity,
        write_out=args.write_out,
        log_samples=args.log_samples,
        gen_kwargs=args.gen_kwargs,
        task_manager=task_manager,
        verbosity=args.verbosity,
        predict_only=args.predict_only,
        random_seed=args.seed[0],
        numpy_random_seed=args.seed[1],
        torch_random_seed=args.seed[2],
        fewshot_random_seed=args.seed[3],
        **request_caching_args,
    )

    if results is not None:
        if args.log_samples:
            samples = results.pop("samples")
        dumped = json.dumps(
            results, indent=2, default=handle_non_serializable, ensure_ascii=False
        )
        if args.show_config:
            print(dumped)

        batch_sizes = ",".join(map(str, results["config"]["batch_sizes"]))

        # Add W&B logging
        if args.wandb_args:
            try:
                wandb_logger.post_init(results)
                wandb_logger.log_eval_result()
                if args.log_samples:
                    wandb_logger.log_eval_samples(samples)
            except Exception as e:
                eval_logger.info(f"Logging to Weights and Biases failed due to {e}")

        evaluation_tracker.save_results_aggregated(
            results=results, samples=samples if args.log_samples else None
        )

        if args.log_samples:
            for task_name, config in results["configs"].items():
                evaluation_tracker.save_results_samples(
                    task_name=task_name, samples=samples[task_name]
                )

        print(
            f"{args.model} ({args.model_args}), gen_kwargs: ({args.gen_kwargs}), limit: {args.limit}, num_fewshot: {args.num_fewshot}, "
            f"batch_size: {args.batch_size}{f' ({batch_sizes})' if batch_sizes else ''}"
        )
        print(make_table(results))
        if "groups" in results:
            print(make_table(results, "groups"))

        if args.wandb_args:
            # Tear down wandb run once all the logging is done.
            wandb_logger.run.finish()

