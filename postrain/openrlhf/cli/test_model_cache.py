import argparse
import os
from datetime import timedelta

import jsonlines
import torch
from torch import distributed as dist
from tqdm import tqdm
from transformers import AutoTokenizer

from openrlhf.datasets import PromptDataset, SFTDataset
from openrlhf.models import IntentionActor, Actor, get_llm_for_sequence_regression
from openrlhf.utils import blending_datasets, get_processor, get_strategy, get_tokenizer
from openrlhf.models.mcts_utils import MCTS

from torch.utils.data import Dataset
from tqdm import tqdm


# boxed instruction
def preprocess_data(data, input_template=None, input_key="input", apply_chat_template=None) -> str:
    if apply_chat_template:
        # format_prompt = [{
        #     "role": "user",
        #     "content": data['Question'],
        # }]
        format_prompt = [
            # {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."}, 
            {
                "role": "user",
                "content": "Please reason step by step, and put your final answer within \\boxed{}.\n\n" + data['Question'],
            }
        ]
        prompt = apply_chat_template(format_prompt, tokenize=False, add_generation_prompt=True)
        answer = data['Answer']
    else:
        prompt = "Please reason step by step, and put your final answer within \\boxed{}.\n\n" + data['Question']
        if input_template:
            prompt = input_template.format(prompt)
        answer = data['Answer']
    return prompt, answer


# # no boxed instructions
# def preprocess_data(data, input_template=None, input_key="input", apply_chat_template=None) -> str:
#     if apply_chat_template:
#         format_prompt = [
#             {
#                 "role": "user",
#                 "content": data['Question'],
#             }
#         ]
#         prompt = apply_chat_template(format_prompt, tokenize=False, add_generation_prompt=True)
#         answer = data['Answer']
#     else:
#         prompt = data['Question']
#         if input_template:
#             prompt = input_template.format(prompt)
#         answer = data['Answer']
#     return prompt, answer


class MathDataset(Dataset):
    """
    Dataset for PPO model

    Args:
        dataset: dataset for PPO model
        tokenizer: tokenizer for PPO model
        max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer,
        strategy,
        input_template=None,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.tokenizer = tokenizer
        self.n_samples_per_prompt = getattr(self.strategy.args, "n_samples_per_prompt", 1)

        # chat_template
        self.input_template = input_template
        input_key = getattr(self.strategy.args, "input_key", None)
        apply_chat_template = getattr(self.strategy.args, "apply_chat_template", False)

        if apply_chat_template:
            apply_chat_template = self.tokenizer.apply_chat_template

        self.prompts = []
        self.raw_prompts = []
        self.answers = []
        for data in tqdm(dataset, desc="Preprocessing data", disable=not self.strategy.is_rank_0()):
            self.raw_prompts.append(data['Question'])
            prompt, answer = preprocess_data(data, input_template, input_key, apply_chat_template)
            self.prompts.append(prompt)
            self.answers.append(answer)

    def __len__(self):
        length = len(self.prompts)
        return length * self.n_samples_per_prompt

    def __getitem__(self, idx):
        return self.prompts[idx // self.n_samples_per_prompt], self.raw_prompts[idx // self.n_samples_per_prompt], self.answers[idx // self.n_samples_per_prompt]


def batch_generate(args):
    # configure strategy
    strategy = get_strategy(args)
    strategy.setup_distributed(timeout=timedelta(minutes=720))

    # configure model
    model = IntentionActor(
        args.pretrain,
        use_flash_attention_2=args.flash_attn,
        bf16=args.bf16,
    )
    # configure tokenizer
    tokenizer = get_tokenizer(args.pretrain, model.model, "left", strategy, use_fast=not args.disable_fast_tokenizer)

    # qmodel = get_llm_for_sequence_regression(
    #     args.q,
    #     "q",
    #     normalize_reward=False,
    #     use_flash_attention_2=args.flash_attn,
    #     bf16=args.bf16,
    #     load_in_4bit=False,
    #     lora_rank=0,
    #     lora_alpha=16,
    #     target_modules=None,
    #     lora_dropout=0.0,
    #     ds_config=None,
    #     value_head_prefix=args.value_head_prefix,
    #     init_value_head=True,
    #     action_size=64,
    # )
    print(model.model.config.vocab_size)
    qmodel = get_llm_for_sequence_regression(
        args.q,
        "q",
        normalize_reward=False,
        use_flash_attention_2=args.flash_attn,
        bf16=args.bf16,
        load_in_4bit=False,
        lora_rank=0,
        lora_alpha=16,
        target_modules=None,
        lora_dropout=0.0,
        ds_config=None,
        action_size=64,
        value_head_prefix="value_head",
        init_value_head=False
    )
    # print(qmodel.lm_head.weight.shape)

    get_tokenizer(args.q, qmodel, "left", strategy, use_fast=not args.disable_fast_tokenizer)
    assert 0

    # prepare models
    model, qmodel = strategy.prepare(model, qmodel)
    model.eval()
    qmodel.eval()

    # tokenizer
    def tokenize_fn(texts):
        batch = tokenizer(
            texts,
            return_tensors="pt",
            add_special_tokens=False,
            max_length=args.prompt_max_len,
            padding=True,
            truncation=True,
        )
        return {k: v.to(torch.cuda.current_device()) for k, v in batch.items()}
    
    # import pandas
    # import os, re
    # df = pandas.read_csv(
    #     os.path.join("/workspace/chengxing/evals/glm-simple-evals/data", "math/math_test.csv")
    # )
    # prompts_data = [row.to_dict() for _, row in df.iterrows()]#[-200:-100]
    import json
    with open("/workspace/chengxing/evals/glm-simple-evals/data/math/math500.json", "r") as f:
        data = json.load(f)
    # start = 100
    # prompts_data = [{"Question": d['problem'], "Answer": d['answer']} for d in data][start:start+args.max_samples]
    prompts_data = [{"Question": d['problem'], "Answer": d['answer']} for d in data]
    if len(prompts_data) // args.max_samples > 0:
        prompts_data = prompts_data[::len(prompts_data) // args.max_samples]

    # answers_data = [row.to_dict()['Answer'] for _, row in df.iterrows()]#[:2]
    N = args.best_of_n
    prompts_dataset = MathDataset(prompts_data, tokenizer, strategy, input_template=args.input_template)
    # prompts = list(prompts_dataset)
    prompts_dataloader = strategy.setup_dataloader(
        prompts_dataset, args.micro_batch_size, True, False, drop_last=False
    )
    pbar = tqdm(
        prompts_dataloader,
        desc="Generating",
        disable=not strategy.is_rank_0(),
    )

    dist.barrier()
    N = args.best_of_n if not args.greedy_intention else 1
    output_dataset = []
    model.model.model.deterministic = args.greedy_intention

    for prompts, rprompts, answers in pbar:
        for prompt, rprompt, answer in zip(prompts, rprompts, answers):
            inputs = tokenizer(prompt, return_tensors='pt')
            inputs = {k: v.to(torch.cuda.current_device()) for k, v in inputs.items()}

            cache = None
            import time
            for i in range(2):
                e = time.time()
                outputs, cache = qmodel(
                    **inputs,
                    use_cache=True,
                    past_key_values=cache,
                )
                print(i+1, time.time() - e)
                inputs["input_ids"] = inputs['input_ids'][:, -1:]
                inputs["attention_mask"] = inputs['attention_mask'][:, -1:]

            break
        break

    





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval_task", type=str, default=None, help="Set to generate_vllm, generate (HF generate) or rm"
    )
    parser.add_argument("--zero_stage", type=int, default=0, help="DeepSpeed ZeRO Stage")
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for deepspeed cli")
    parser.add_argument("--bf16", action="store_true", default=False, help="Enable bfloat16 for deepspeed")
    parser.add_argument("--flash_attn", action="store_true", default=False, help="Enable FlashAtten2")
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False)
    parser.add_argument("--micro_batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--remote_rm_url", type=str, default=None, help="remote RM API")

    # Models
    parser.add_argument("--pretrain", type=str, default=None, help="HF pretrain model name or path")
    parser.add_argument("--q", type=str, default=None, help="HF q model name or path")
    parser.add_argument(
        "--value_head_prefix", type=str, default="value_head", help="value_head prefix for Reward Model"
    )

    # Custom dataset
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--dataset_probs", type=str, default="1.0")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--input_key", type=str, default="input", help="JSON dataset key")
    parser.add_argument("--output_key", type=str, default="output", help="JSON dataset key")
    parser.add_argument(
        "--apply_chat_template", action="store_true", default=False, help="HF tokenizer apply_chat_template"
    )
    parser.add_argument("--input_template", type=str, default=None)
    parser.add_argument("--max_len", type=int, default=2048, help="Max tokens for the samples")
    parser.add_argument("--max_samples", type=int, default=1e8, help="Max number of samples")
    parser.add_argument("--output_path", type=str, default=None, help="Output JSON data path")

    # For generation
    parser.add_argument("--prompt_max_len", type=int, default=1024, help="Max tokens for prompt")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Max new tokens in generation")
    parser.add_argument("--greedy_sampling", action="store_true", default=False, help="Use Greedy sampling")
    parser.add_argument("--greedy_intention", action="store_true", default=False, help="Use Greedy sampling")
    parser.add_argument("--top_p", type=float, default=1.0, help="top_p for Sampling")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature for Sampling")
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--best_of_n", type=int, default=1, help="Number of responses to generate per prompt")
    parser.add_argument(
        "--post_processor",
        type=str,
        default=None,
        help="set to rs (Rejection Sampling), csft (Conditional SFT), iter_dpo (Iterative DPO) or None",
    )
    # For vllm
    parser.add_argument("--tp_size", type=int, default=torch.cuda.device_count())
    parser.add_argument("--max_num_seqs", type=int, default=256)
    parser.add_argument("--enable_prefix_caching", action="store_true", default=False)

    # For Iterative generation and Rejection Sampling
    parser.add_argument(
        "--iter",
        type=int,
        default=None,
        help="Used to slice the datasets in range iter * rollout_batch_size: (iter + 1) * rollout_batch_size",
    )
    parser.add_argument("--rollout_batch_size", type=int, default=2048, help="Number of samples to generate")

    # For Conditional SFT
    parser.add_argument("--normalize_reward", action="store_true", default=False, help="Enable Reward Normazation")
    parser.add_argument("--reward_template", type=str, default=None)
    parser.add_argument("--enable_csft", action="store_true", default=False)
    parser.add_argument("--csft_prompt", type=str, default="<rm_score>: 5.00", help="Conditional SFT prompt")

    args = parser.parse_args()
    if args.eval_task and args.eval_task == "generate":
        with torch.no_grad():
            batch_generate(args)
