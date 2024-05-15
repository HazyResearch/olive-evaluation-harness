from lm_eval.config import EvalConfig
output_dir = "/var/cr01_data/sabri/code/olive/olive-evaluation-harness/outputs"

config = EvalConfig(
    model="hf-mod",
    model_args="pretrained=meta-llama/Meta-Llama-3-8B-Instruct,layer_swaps=16:0",
    tasks="triviaqa",
    device="cuda:0",
    batch_size="8",
    num_fewshot=1,
    output_path="./",
    wandb=True,
    wandb_args="entity=hazy-research,project=olive-eval,job_type=eval"
)

configs = [config]

