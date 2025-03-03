import argparse
import math
import os
from datetime import datetime

from transformers.trainer import get_scheduler

from openrlhf.datasets import SFTDataset
from openrlhf.models import IntentionActor
from openrlhf.trainer import SFTITrainer
from openrlhf.utils import blending_datasets, get_strategy, get_tokenizer

import torch.nn as nn
def mark_only_param_as_trainable(model: nn.Module, bias = ["policy"]) -> None:
    trainbale_list = []
    for n, p in model.named_parameters():
        req_g = False
        for name in bias:
            if name in n:
                req_g = True
                trainbale_list.append(n)
                break
                
        p.requires_grad = req_g
    print("req_g: ", trainbale_list)


def train(args):
    # configure strategy
    strategy = get_strategy(args)
    strategy.setup_distributed()

    # configure model
    # load huggingface model
    model = IntentionActor(
        args.pretrain,
        use_flash_attention_2=args.flash_attn,
        bf16=args.bf16,
        load_in_4bit=args.load_in_4bit,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules,
        lora_dropout=args.lora_dropout,
        ds_config=strategy.get_ds_train_config(is_actor=True),
        packing_samples=args.packing_samples,
    )
    idm_model = IntentionActor(
        args.idm,
        use_flash_attention_2=args.flash_attn,
        bf16=args.bf16,
        load_in_4bit=args.load_in_4bit,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules,
        lora_dropout=args.lora_dropout,
        ds_config=strategy.get_ds_train_config(is_actor=True),
        packing_samples=args.packing_samples,
    )
    # configure tokenizer
    tokenizer = get_tokenizer(args.pretrain, model.model, "right", strategy, use_fast=not args.disable_fast_tokenizer)
    strategy.print(model)

    model.model.model.action_layers.load_state_dict(idm_model.model.model.action_layers.state_dict())
    model.model.model.action_norm.load_state_dict(idm_model.model.model.action_norm.state_dict())
    model.model.model.action_head.load_state_dict(idm_model.model.model.action_head.state_dict())

    # prepare models
    model = strategy.prepare(model)

    # save model checkpoint after fitting on only rank0
    strategy.save_model(model.model, tokenizer, args.save_path)


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
    parser.add_argument("--idm", type=str, default=None)
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
    parser.add_argument("--additional_output_key", type=str, default=None, help="JSON dataset key")
    parser.add_argument("--input_template", type=str, default="User: {}\nAssistant: ")
    parser.add_argument(
        "--apply_chat_template", action="store_true", default=False, help="Use HF tokenizer chat template"
    )
    parser.add_argument("--tokenizer_chat_template", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=1e8, help="Max number of samples")
    parser.add_argument("--max_len", type=int, default=2048, help="Max tokens for the samples")

    parser.add_argument(
        "--train_policy", action="store_true", help="only train policy in intention llama"
    )
    parser.add_argument(
        "--train_model", action="store_true", help="train both inverse dynamic model and world model in intention llama"
    )

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
