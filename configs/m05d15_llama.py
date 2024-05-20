from lm_eval.config import EvalConfig
output_dir = "/var/cr01_data/sabri/code/olive/olive-evaluation-harness/outputs"

configs = []
for layer in [0, 4, 8, 12, 16, 20, 24, 28]:
    config = EvalConfig(
        model="hf-mod",
        model_args=f"pretrained=meta-llama/Meta-Llama-3-8B-Instruct,layer_swaps={layer}:{layer + 1}",
        gen_kwargs="max_new_tokens=12",
        tasks="triviaqa",
        device="cuda:0",
        batch_size="8",
        num_fewshot=0,
        output_path="./",
        wandb=True,
        wandb_args="entity=hazy-research,project=olive-eval,job_type=eval",
        
        run_id=f"swap_layers_layer{layer}",
    )
    configs.append(config)


