import argparse
import math
import os
from datetime import datetime

from transformers.trainer import get_scheduler

from openrlhf.datasets import SFTDataset
from openrlhf.models import Actor, get_llm_for_sequence_regression
from openrlhf.trainer import QTrainer
from openrlhf.utils import blending_datasets, get_strategy, get_tokenizer

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
                "content": "Please reason step by step, and put your final answer within \\boxed{}.\n\n" + data['question'],
            }
        ]
        prompt = apply_chat_template(format_prompt, tokenize=False, add_generation_prompt=True)
        answer = data['response']
    else:
        prompt = "Please reason step by step, and put your final answer within \\boxed{}.\n\n" + data['question']
        if input_template:
            prompt = input_template.format(prompt)
        answer = data['response']
    return prompt, answer

import torch
from openrlhf.datasets.utils import zero_pad_sequences

class QDataset(Dataset):
    """
    Dataset for SFT model

    Args:
        dataset: dataset for SFT model
        tokenizer: tokenizer for SFT model
        max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer,
        max_length: int,
        strategy,
        input_template=None,
        pretrain_mode=False,
        num_processors=8,  # Specify the number of processors you want to use
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.pretrain_mode = pretrain_mode
        self.max_length = max_length

        # chat template
        self.input_template = input_template
        self.input_key = getattr(self.strategy.args, "input_key", None)
        self.output_key = getattr(self.strategy.args, "output_key", None)
        self.apply_chat_template = getattr(self.strategy.args, "apply_chat_template", False)

        if self.apply_chat_template:
            self.apply_chat_template = self.tokenizer.apply_chat_template
            tokenizer_chat_template = getattr(self.strategy.args, "tokenizer_chat_template", None)
            if tokenizer_chat_template:
                self.tokenizer.chat_template = tokenizer_chat_template

        # Parallel loading datasets
        self.IGNORE_INDEX = -100
        self.prompts = []
        self.responses = []
        self.prompt_ids_lens = []
        self.rewards = []
        # self.action_idxs = []
        for data in dataset:
            output = self.process_data(data)
            if output['prompt'] is None:
                continue
            self.prompts.append(output['prompt'])
            self.responses.append(output['response'])
            self.prompt_ids_lens.append(output['prompt_ids_len'])

            self.rewards.append(data['reward'])
            # self.action_idxs.append(data['action'])

    def process_data(self, data):
        prompt, response = preprocess_data(
            data,
            None if self.pretrain_mode else self.input_template,
            self.input_key,
            apply_chat_template=None if self.pretrain_mode else self.apply_chat_template,
        )
        if not self.pretrain_mode:
            prompt_token = self.tokenizer(
                prompt,
                max_length=self.max_length,
                padding=False,
                truncation=True,
                return_tensors="pt",
                add_special_tokens=False,
            )
            prompt_ids_len = prompt_token["attention_mask"].int().sum().item()

            # filter the sample whose length is greater than max_length (2 for answer length)
            if not prompt or not response or prompt_ids_len >= self.max_length - 2:
                prompt = None
        else:
            prompt_ids_len = 0

        return {"prompt": prompt, "response": response, "prompt_ids_len": prompt_ids_len}

    def __len__(self):
        length = len(self.prompts)
        return length

    def __getitem__(self, idx):
        prompt_ids_len = self.prompt_ids_lens[idx]
        prompt = self.prompts[idx]
        response = self.responses[idx]

        reward = torch.FloatTensor([self.rewards[idx]])
        # action = self.action_idxs[idx]

        if not self.pretrain_mode:
            text = (prompt + response).rstrip("\n")
            if not text.endswith(self.tokenizer.eos_token):
                text += " " + self.tokenizer.eos_token
        else:
            text = prompt

        input_token = self.tokenizer(
            text,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )

        # process reward
        reward_seq = torch.zeros((input_token["input_ids"].shape[0], input_token["input_ids"].shape[1])).float()
        attention_mask = input_token["attention_mask"]
        eos_indices = attention_mask.size(1) - attention_mask.long().fliplr().argmax(dim=1, keepdim=True).clamp(min=1)
        reward_seq = reward_seq.scatter_(dim=1, index=eos_indices - 1, src=reward.unsqueeze(1))

        # process action
        # action_seq = torch.zeros_like(input_token["input_ids"]) + self.IGNORE_INDEX
        # end_idx = eos_indices[0, 0]
        # start_idx = prompt_ids_len
        # max_index = min(self.max_length, end_idx-start_idx+1)
        # if max_index > len(action):
        #     action += [0 for _ in range(max_index - len(action))]
        #     # print("add")
        # action_seq[:, start_idx-1:end_idx] = torch.LongTensor(action)[:max_index].type(action_seq.dtype)

        if not self.pretrain_mode:
            # to avoid EOS_token truncation
            input_token["input_ids"][0][-1] = self.tokenizer.eos_token_id
            input_token["attention_mask"][0][-1] = True
        info = {"input": prompt, "output": response, "input_length": input_token["attention_mask"].int().sum().item()}

        return prompt_ids_len, input_token["input_ids"], input_token["attention_mask"], reward_seq, info

    def collate_fn(self, item_list):
        prompt_ids_lens = []
        input_ids = []
        attention_masks = []
        rewards = []
        # actions = []
        infos = {"input": [], "output": []}

        for prompt_ids_len, input_id, attention_mask, reward, info in item_list:
            prompt_ids_lens.append(prompt_ids_len)
            input_ids.append(input_id)
            attention_masks.append(attention_mask)
            rewards.append(reward)
            # actions.append(action)
            infos["input"].append(info["input"])
            infos["output"].append(info["output"])

        input_ids = zero_pad_sequences(input_ids, "right", self.tokenizer.pad_token_id)
        attention_masks = zero_pad_sequences(attention_masks, "right")
        # actions = zero_pad_sequences(actions, "right", self.IGNORE_INDEX)
        rewards = zero_pad_sequences(rewards, "right", 0)
        return prompt_ids_lens, input_ids, attention_masks, rewards, infos


def train(args):
    # configure strategy
    strategy = get_strategy(args)
    strategy.setup_distributed()

    # configure model
    # load huggingface model
    model = Actor(
        args.pretrain,
        use_flash_attention_2=args.flash_attn,
        bf16=args.bf16,
        load_in_4bit=args.load_in_4bit,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules,
        lora_dropout=args.lora_dropout,
        packing_samples=args.packing_samples,
    )

    # configure tokenizer
    tokenizer = get_tokenizer(args.pretrain, model.model, "right", strategy, use_fast=not args.disable_fast_tokenizer)
    strategy.print(model)

    qmodel = get_llm_for_sequence_regression(
        args.pretrain,
        "q",
        normalize_reward=False,
        use_flash_attention_2=args.flash_attn,
        bf16=args.bf16,
        load_in_4bit=args.load_in_4bit,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules,
        lora_dropout=args.lora_dropout,
        ds_config=strategy.get_ds_train_config(is_actor=False),
        value_head_prefix=args.value_head_prefix,
        init_lm_head=True,
        action_size=model.model.config.vocab_size,
    )
    target_qmodel = get_llm_for_sequence_regression(
        args.pretrain,
        "q",
        normalize_reward=False,
        use_flash_attention_2=args.flash_attn,
        bf16=args.bf16,
        load_in_4bit=args.load_in_4bit,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules,
        lora_dropout=args.lora_dropout,
        ds_config=strategy.get_ds_train_config(is_actor=False),
        value_head_prefix=args.value_head_prefix,
        init_lm_head=True,
        action_size=model.model.config.vocab_size,
    )
    target_qmodel.lm_head.load_state_dict(qmodel.lm_head.state_dict())
    target_qmodel.eval()
    del model

    # gradient_checkpointing
    if args.gradient_checkpointing:
        qmodel.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": args.gradient_checkpointing_use_reentrant}
        )

    # configure optimizer
    optim = strategy.create_optimizer(qmodel, lr=args.learning_rate, betas=args.adam_betas, weight_decay=args.l2)

    # prepare for data and dataset
    import json
    path_math = "/workspace/chengxing/intentionrl/openrlhf/checkpoint/NuminaMath-CoT/math-best-of-n-s0-dt16-action-math-train/output_500_pro.json"
    # path_math = "/workspace/chengxing/intentionrl/openrlhf/checkpoint/NuminaMath-CoT/math-best-of-n-s0-dt16-action/output_500_pro.json"
    with open(path_math, "r") as f:
        data = json.load(f)
    num_samples = 0
    train_data = []
    for d in data:
        for response, action_idx, score in zip(d["assistant"], d["action_idx"], d["score"]):
            # print(len(action_idx))
            train_data.append({
                "question": d["user"],
                "response": response,
                "action": action_idx,
                "reward": score,
            })
            num_samples += 1
        if num_samples >= args.max_samples:
            break
    train_dataset = QDataset(train_data, tokenizer, args.max_len, strategy, input_template=args.input_template)
    eval_dataset = train_dataset

    # prepare dataloader
    train_dataloader = strategy.setup_dataloader(
        train_dataset,
        args.micro_train_batch_size,
        True,
        True,
        train_dataset.packing_collate_fn if args.packing_samples else train_dataset.collate_fn,
    )
    eval_dataloader = strategy.setup_dataloader(
        eval_dataset,
        args.micro_train_batch_size,
        True,
        False,
        eval_dataset.packing_collate_fn if args.packing_samples else eval_dataset.collate_fn,
    )

    # scheduler
    num_update_steps_per_epoch = len(train_dataset) // args.train_batch_size
    max_steps = math.ceil(args.max_epochs * num_update_steps_per_epoch)

    scheduler = get_scheduler(
        args.lr_scheduler,
        optim,
        num_warmup_steps=math.ceil(max_steps * 0.03),
        num_training_steps=max_steps,
        scheduler_specific_kwargs={"min_lr": args.learning_rate * 0.1},
    )

    # prepare models
    (qmodel, optim, scheduler), target_qmodel = strategy.prepare((qmodel, optim, scheduler), target_qmodel)

    # load checkpoint
    consumed_samples = 0
    os.makedirs(args.save_path, exist_ok=True)

    # target_qmodel.reset_target(qmodel)
    # for prompts_id_lens, inputs, attention_masks, rewards, infos in eval_dataloader:
    #     a = 1
        # print(inputs.shape, attention_masks.shape, rewards.shape, actions.shape)
        # print(inputs.dtype, attention_masks.dtype, rewards.dtype, actions.dtype)

    # configure Trainer
    trainer = QTrainer(
        model=qmodel,
        target_model=target_qmodel,
        strategy=strategy,
        optim=optim,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        scheduler=scheduler,
        max_norm=args.max_norm,
        pretrain_mode=args.pretrain_mode,
        batch_size=args.train_batch_size,
        max_epochs=args.max_epochs,
        tokenizer=tokenizer,
    )

    # trainer.fit(args, consumed_samples, num_update_steps_per_epoch)

    # save model checkpoint after fitting on only rank0
    strategy.save_model(qmodel, tokenizer, args.save_path)

    if args.local_rank == 0:
        trainer.save_logger(args.save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Checkpoint
    parser.add_argument("--save_path", type=str, default="./ckpt")
    parser.add_argument("--save_steps", type=int, default=-1)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=-1)
    parser.add_argument("--ckpt_path", type=str, default="./ckpt/checkpoints_sft")
    parser.add_argument("--max_ckpt_num", type=int, default=3)
    parser.add_argument("--max_ckpt_mem", type=int, default=1e8)
    parser.add_argument("--load_checkpoint", action="store_true", default=False)

    # DeepSpeed
    parser.add_argument("--micro_train_batch_size", type=int, default=8, help="batch size per GPU")
    parser.add_argument("--train_batch_size", type=int, default=128, help="Global training batch size")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for deepspeed")
    parser.add_argument("--zero_stage", type=int, default=2, help="DeepSpeed ZeRO stage")
    parser.add_argument("--bf16", action="store_true", default=False, help="Enable bfloat16")
    parser.add_argument("--zpg", type=int, default=1, help="ZeRO++ max partition size")
    parser.add_argument("--adam_offload", action="store_true", default=False, help="Offload Adam Optimizer")
    parser.add_argument("--flash_attn", action="store_true", default=False, help="Enable FlashAttention2")
    parser.add_argument("--grad_accum_dtype", type=str, default=None, help="Adam grad accum data type")
    parser.add_argument("--disable_trace_cache", action="store_true", default=False)
    parser.add_argument("--gradient_checkpointing_use_reentrant", action="store_true", default=False)
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False)

    # SFT
    parser.add_argument("--max_epochs", type=int, default=2)
    parser.add_argument("--aux_loss_coef", type=float, default=0, help="MoE balancing loss")
    parser.add_argument("--pretrain", type=str, default=None)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--pretrain_mode", action="store_true", default=False, help="Use pretrain loss")
    parser.add_argument("--lr_scheduler", type=str, default="cosine_with_min_lr")
    parser.add_argument("--l2", type=float, default=0, help="weight decay loss")
    parser.add_argument("--adam_betas", type=float, nargs=2, default=(0.9, 0.95), help="Betas for Adam optimizer")

    # LoRA
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--lora_rank", type=int, default=0)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--target_modules", type=str, nargs="*", default="all-linear")
    parser.add_argument("--lora_dropout", type=float, default=0)

    # packing SFT samples without CrossAttention
    parser.add_argument("--packing_samples", action="store_true", default=False)

    # custom dataset
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--dataset_probs", type=str, default="1.0", help="sampling probs for datasets")
    parser.add_argument("--train_split", type=str, default="train", help="train split of the HF dataset")
    parser.add_argument("--eval_split", type=str, default="test", help="test split of the dataset")

    parser.add_argument("--input_key", type=str, default="input", help="JSON dataset key")
    parser.add_argument("--output_key", type=str, default=None, help="JSON dataset key")
    parser.add_argument("--input_template", type=str, default="User: {}\nAssistant: ")
    parser.add_argument(
        "--apply_chat_template", action="store_true", default=False, help="Use HF tokenizer chat template"
    )
    parser.add_argument("--tokenizer_chat_template", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=1e8, help="Max number of samples")
    parser.add_argument("--max_len", type=int, default=2048, help="Max tokens for the samples")

    parser.add_argument("--value_head_prefix", type=str, default="lm_head")

    # wandb parameters
    parser.add_argument("--use_wandb", type=str, default=None)
    parser.add_argument("--wandb_org", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="openrlhf_train_sft")
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="sft_%s" % datetime.now().strftime("%m%dT%H:%M"),
    )

    args = parser.parse_args()

    if args.input_template and not "{}" in args.input_template:
        print("[Warning] {} not in args.input_template, set to None")
        args.input_template = None

    train(args)
