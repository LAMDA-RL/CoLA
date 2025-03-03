import math
from abc import ABC

import torch
from torch import nn
from torch.optim import Optimizer
from tqdm import tqdm
from transformers.trainer import get_scheduler

from openrlhf.datasets import SFTDataset
from openrlhf.models import GPTLMLoss
from openrlhf.utils.distributed_sampler import DistributedSampler
from openrlhf.utils.logger import Logger


class QTrainer(ABC):
    """
        Trainer to use while training reward model.

    Args:
        model (torch.nn.Module): the model to train
        strategy (Strategy): the strategy to use for training
        optim(Optimizer): the optimizer to use for training
        train_dataset (RewardDataset): the dataset to use for training
        eval_dataset (RewardDataset): the dataset to use for evaluation
        batch_size (int, defaults to 1): the batch size while training
        max_epochs (int, defaults to 2): the number of epochs to train
        optim_kwargs (dict, defaults to {'lr':1e-4}): the kwargs to use while initializing optimizer
    """

    def __init__(
        self,
        model,
        target_model,
        strategy,
        optim: Optimizer,
        train_dataloader,
        eval_dataloader,
        scheduler,
        max_norm: float = 1,
        pretrain_mode: bool = False,
        batch_size: int = 1,
        max_epochs: int = 2,
        tokenizer=None,
        stage=1,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.epochs = max_epochs
        self.batch_size = batch_size
        self.max_norm = max_norm
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.scheduler = scheduler
        self.pretrain_mode = pretrain_mode
        self.model = model
        self.target_model = target_model
        self.tokenizer = tokenizer
        self.optimizer = optim
        self.args = strategy.args
        self.stage = stage

        # label noise
        # self.add_label_noise = self.args.label_noise
        # self.label_noise_ratio = self.args.label_noise_ratio

        self.loss_fn = GPTLMLoss()
        self.IGNORE_INDEX = -100
        self.gamma = 0.99

        # Mixtral 8*7b
        self.aux_loss = self.args.aux_loss_coef > 1e-8

        # packing samples
        self.packing_samples = strategy.args.packing_samples

        # wandb setting
        self._wandb = None
        if self.strategy.args.use_wandb and self.strategy.is_rank_0():
            import wandb

            self._wandb = wandb
            if not wandb.api.api_key:
                wandb.login(key=strategy.args.use_wandb)
            wandb.init(
                entity=strategy.args.wandb_org,
                project=strategy.args.wandb_project,
                group=strategy.args.wandb_group,
                name=strategy.args.wandb_run_name,
                config=strategy.args.__dict__,
                reinit=True,
            )

            wandb.define_metric("train/global_step")
            wandb.define_metric("train/*", step_metric="train/global_step", step_sync=True)
            wandb.define_metric("eval/global_step")
            wandb.define_metric("eval/*", step_metric="eval/global_step", step_sync=True)
        
        self.eval_logger = Logger()
        self.train_logger = Logger()

    def fit(self, args, consumed_samples=0, num_update_steps_per_epoch=None):
        # get eval and save steps
        if args.eval_steps == -1:
            args.eval_steps = num_update_steps_per_epoch  # Evaluate once per epoch
        if args.save_steps == -1:
            args.save_steps = float("inf")  # do not save ckpt

        # Restore step and start_epoch
        step = consumed_samples // args.train_batch_size * self.strategy.accumulated_gradient + 1
        start_epoch = consumed_samples // args.train_batch_size // num_update_steps_per_epoch
        consumed_samples = consumed_samples % (num_update_steps_per_epoch * args.train_batch_size)

        epoch_bar = tqdm(
            range(start_epoch, self.epochs),
            desc="Train epoch",
            disable=not self.strategy.is_rank_0(),
        )
        for epoch in range(start_epoch, self.epochs):
            if isinstance(self.train_dataloader.sampler, DistributedSampler):
                self.train_dataloader.sampler.set_epoch(
                    epoch, consumed_samples=0 if epoch > start_epoch else consumed_samples
                )

            step_bar = tqdm(
                range(self.train_dataloader.__len__()),
                desc="Train step of epoch %d" % epoch,
                disable=not self.strategy.is_rank_0(),
            )

            # train
            self.model.train()
            loss_mean = 0
            for prompts_id_lens, inputs, attention_masks, rewards, actions, infos in self.train_dataloader:
                inputs = inputs.to(torch.cuda.current_device()).squeeze(1)
                attention_mask = attention_masks.to(torch.cuda.current_device()).squeeze(1)
                rewards = rewards.to(torch.cuda.current_device()).squeeze(1)
                actions = actions.to(torch.cuda.current_device()).squeeze(1)
                
                # r = rewards[0].cpu().numpy().tolist()
                # a = actions[0].cpu().numpy().tolist()
                # idx = 0
                # for i, reward in enumerate(r):
                #     if reward != 0.0:
                #         idx = i
                #         break
                # print(a[:idx+1])
                # print(r[:idx+1])
                # assert 0

                time_mask = (actions != self.IGNORE_INDEX)
                actions[~time_mask] = 0
                outputs = self.model(inputs, attention_mask=attention_mask)  # bs, lens, 64
                max_actions = outputs.argmax(dim=-1)
                with torch.no_grad():
                    target_outputs = self.target_model(inputs, attention_mask=attention_mask)  # bs, lens, 64
                    target_qvalues = torch.gather(target_outputs, dim=-1, index=max_actions.unsqueeze(-1)).squeeze(-1)
                    target_values = rewards[:, :-1] + self.gamma * target_qvalues[:, 1:] * time_mask[:, 1:]

                select_qvalues = torch.gather(outputs, dim=-1, index=actions.unsqueeze(-1)).squeeze(-1)[:, :-1]
                loss_step = ((select_qvalues - target_values) ** 2) * time_mask[:, :-1]
                loss = loss_step.sum() / time_mask[:, :-1].sum()

                self.strategy.backward(loss, self.model, self.optimizer)
                self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)

                # print((rewards[:, :-1] * time_mask[:, :-1]).sum())

                loss_mean = loss_mean * 0.9 + 0.1 * loss.item()
                logs_dict = {
                    "loss_mean": loss_mean,
                    "loss_max": loss_step.max().item(),
                    "num_sample": time_mask.sum().float().item(),
                    "lr": self.scheduler.get_last_lr()[0],
                }
                # if self.aux_loss:
                #     logs_dict["aux_loss"] = aux_loss.item()
                # step bar
                logs_dict = self.strategy.all_reduce(logs_dict)
                step_bar.set_postfix(logs_dict)
                step_bar.update()

                self.train_logger.add(logs_dict)

                if step % 50 == 0:
                    self.target_model.reset_target(self.model)

                # logs/checkpoints/evaluation
                if step % self.strategy.accumulated_gradient == 0:
                    global_step = step // self.strategy.accumulated_gradient
                    client_states = {"consumed_samples": global_step * args.train_batch_size}
                    self.save_logs_and_checkpoints(args, global_step, step_bar, logs_dict, client_states)

                step += 1

            epoch_bar.update()

    # logs/checkpoints/evaluation
    def save_logs_and_checkpoints(self, args, global_step, step_bar, logs_dict={}, client_states={}):
        if global_step % args.logging_steps == 0:
            # wandb
            if self._wandb is not None and self.strategy.is_rank_0():
                logs = {"train/%s" % k: v for k, v in {**logs_dict, "global_step": global_step}.items()}
                self._wandb.log(logs)

        # eval
        if global_step % args.eval_steps == 0:
            self.evaluate(self.eval_dataloader, global_step)
        # save ckpt
        # TODO: save best model on dev, use loss/perplexity on whole dev dataset as metric
        if global_step % args.save_steps == 0:
            tag = f"global_step{global_step}"
            self.strategy.save_ckpt(
                self.model.model, args.ckpt_path, tag, args.max_ckpt_num, args.max_ckpt_mem, client_states
            )

    def evaluate(self, eval_dataloader, steps=0):
        pass

    def save_logger(self, path):
        import os
        self.train_logger.save(os.path.join(path, "train.npy"))
        # self.eval_logger.save(os.path.join(path, "eval.npy"))
