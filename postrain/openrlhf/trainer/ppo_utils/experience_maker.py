import logging
import time
from abc import ABC
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import ray
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from openrlhf.models.actor import Actor
from openrlhf.models.utils import compute_reward, masked_mean, compute_reward_remax
from openrlhf.utils.logging_utils import init_logger
from openrlhf.utils.remote_rm_utils import remote_rm_fn, remote_rm_fn_ray
import re
logger = init_logger(__name__)

from openrlhf.trainer.ppo_utils.math_eval_vine import grade_answer as math_reward
ANSWER_PATTERN = 'boxed{([\s\S]*)}'

@dataclass
class Experience:
    """Experience is a batch of data.
    These data should have the the sequence length and number of actions.
    Left padding for sequences is applied.

    Shapes of each tensor:
    sequences: (B, S)
    action_log_probs: (B, A)
    values: (B, A)
    returns: (B, A)
    advatanges: (B, A)
    attention_mask: (B, S)
    action_mask: (B, A)

    "A" is the number of actions.
    """

    sequences: torch.Tensor
    action_log_probs: torch.Tensor
    values: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    info: Optional[dict]

    @torch.no_grad()
    def to_device(self, device: torch.device) -> None:
        self.sequences = self.sequences.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.values = self.values.to(device)
        self.returns = self.returns.to(device)
        self.advantages = self.advantages.to(device)
        if self.attention_mask is not None:
            self.attention_mask = self.attention_mask.to(device)
        if self.action_mask is not None:
            self.action_mask = self.action_mask.to(device)

    def pin_memory(self):
        self.sequences = self.sequences.pin_memory()
        self.action_log_probs = self.action_log_probs.pin_memory()
        self.values = self.values.pin_memory()
        self.returns = self.returns.pin_memory()
        self.advantages = self.advantages.pin_memory()
        if self.attention_mask is not None:
            self.attention_mask = self.attention_mask.pin_memory()
        if self.action_mask is not None:
            self.action_mask = self.action_mask.pin_memory()
        return self


@dataclass
class ExperienceWithAction:
    """Experience is a batch of data.
    These data should have the the sequence length and number of actions.
    Left padding for sequences is applied.

    Shapes of each tensor:
    sequences: (B, S)
    action_log_probs: (B, A)
    values: (B, A)
    returns: (B, A)
    advatanges: (B, A)
    attention_mask: (B, S)
    action_mask: (B, A)

    "A" is the number of actions.
    """

    sequences: torch.Tensor
    action_sequences: torch.Tensor
    # policy_embeds: torch.Tensor
    action_log_probs: torch.Tensor
    values: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    info: Optional[dict]

    @torch.no_grad()
    def to_device(self, device: torch.device) -> None:
        self.sequences = self.sequences.to(device)
        self.action_sequences = self.action_sequences.to(device)
        # self.policy_embeds = self.policy_embeds.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.values = self.values.to(device)
        self.returns = self.returns.to(device)
        self.advantages = self.advantages.to(device)
        if self.attention_mask is not None:
            self.attention_mask = self.attention_mask.to(device)
        if self.action_mask is not None:
            self.action_mask = self.action_mask.to(device)

    def pin_memory(self):
        self.sequences = self.sequences.pin_memory()
        self.action_sequences = self.action_sequences.pin_memory()
        # self.policy_embeds = self.policy_embeds.pin_memory()
        self.action_log_probs = self.action_log_probs.pin_memory()
        self.values = self.values.pin_memory()
        self.returns = self.returns.pin_memory()
        self.advantages = self.advantages.pin_memory()
        if self.attention_mask is not None:
            self.attention_mask = self.attention_mask.pin_memory()
        if self.action_mask is not None:
            self.action_mask = self.action_mask.pin_memory()
        return self


class ExperienceMaker(ABC):
    """
    Naive experience maker.
    """

    def __init__(
        self,
        actor: Actor,
        reward_model: nn.Module,
        tokenizer,
        prompt_max_len: int,
        kl_controller,
        strategy=None,
        remote_rm_url: str = None,
        reward_fn=None,
    ) -> None:
        super().__init__()
        self.actor = actor
        self.reward_model = reward_model
        self.remote_rm_url = remote_rm_url
        self.tokenizer = tokenizer
        self.prompt_max_len = prompt_max_len
        self.kl_ctl = kl_controller
        self.strategy = strategy
        self.reward_fn = reward_fn

    # tokenizer
    def tokenize_fn(self, texts, max_length, device):
        batch = self.tokenizer(
            texts,
            return_tensors="pt",
            add_special_tokens=False,
            max_length=max_length,
            padding=True,
            truncation=True,
        )
        return {k: v.to(device) for k, v in batch.items()}

    @torch.no_grad()
    def make_experience(self, prompts: Union[str, List[str]], **generate_kwargs):
        self.actor.eval()
        if self.reward_model is not None:
            self.reward_model.eval()

        # generate seq
        inputs = self.tokenize_fn(prompts, self.prompt_max_len, device="cuda")
        sequences, attention_mask, action_mask = self.actor.generate(**inputs, **generate_kwargs)
        num_actions = action_mask.size(1)

        # log probs
        action_log_probs = self.actor(sequences, num_actions, attention_mask)

        # rewards
        if self.remote_rm_url is not None:
            # remote RM
            queries = self.tokenizer.batch_decode(sequences.cpu(), skip_special_tokens=False)
            r = remote_rm_fn(self.remote_rm_url, queries=queries).to(device=action_log_probs.device)
        else:
            # local RM
            r = self.reward_model(sequences, attention_mask)

        info = {
            "reward": r,
            "response_length": action_mask.float().sum(dim=-1),
            "total_length": attention_mask.float().sum(dim=-1),
        }

        return info

    @torch.no_grad()
    def get_advantages_and_returns(
        self,
        values: torch.Tensor,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
        gamma: float,
        lambd: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Function that computes advantages and returns from rewards and values.
        Calculated as in the original PPO paper: https://arxiv.org/abs/1707.06347
        Note that rewards may include a KL divergence loss term.

        Advantages looks like this:
        Adv1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
              - V1 + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Returns looks like this:
        Ret1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
                   + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Input:
        - values: Tensor of shape (batch_size, response_size)
        - rewards: Tensor of shape (batch_size, response_size)

        Output:
        - advantages: Tensor of shape (batch_size, response_size)
        - returns: Tensor of shape (batch_size, response_size)
        """
        lastgaelam = 0
        advantages_reversed = []
        response_length = rewards.size(1)

        # Mask invalid responses
        values = action_mask * values
        rewards = action_mask * rewards

        for t in reversed(range(response_length)):
            nextvalues = values[:, t + 1] if t < response_length - 1 else 0.0
            delta = rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lambd * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values
        return advantages.detach(), returns



class NaiveExperienceMaker(ABC):
    """
    Naive experience maker.
    """

    def __init__(
        self,
        actor: Actor,
        critic: nn.Module,
        reward_model: nn.Module,
        initial_model: Actor,
        tokenizer,
        prompt_max_len: int,
        kl_controller,
        strategy=None,
        remote_rm_url: str = None,
        reward_fn=None,
    ) -> None:
        super().__init__()
        self.actor = actor
        self.critic = critic
        self.reward_model = reward_model
        self.remote_rm_url = remote_rm_url
        self.initial_model = initial_model
        self.tokenizer = tokenizer
        self.prompt_max_len = prompt_max_len
        self.kl_ctl = kl_controller
        self.strategy = strategy
        self.reward_fn = reward_fn

    # tokenizer
    def tokenize_fn(self, texts, max_length, device):
        batch = self.tokenizer(
            texts,
            return_tensors="pt",
            add_special_tokens=False,
            max_length=max_length,
            padding=True,
            truncation=True,
        )
        return {k: v.to(device) for k, v in batch.items()}

    @torch.no_grad()
    def make_experience(self, prompts: Union[str, List[str]], **generate_kwargs) -> Experience:
        self.actor.eval()
        self.critic.eval()
        self.initial_model.eval()
        if self.reward_model is not None:
            self.reward_model.eval()

        # generate seq
        inputs = self.tokenize_fn(prompts, self.prompt_max_len, device="cuda")
        sequences, attention_mask, action_mask = self.actor.generate(**inputs, **generate_kwargs)
        num_actions = action_mask.size(1)

        # log probs
        action_log_probs = self.actor(sequences, num_actions, attention_mask)

        # init log probs
        base_action_log_probs = self.initial_model(sequences, num_actions, attention_mask)

        # values
        value = self.critic(sequences, num_actions, attention_mask)

        # rewards
        if self.remote_rm_url is not None:
            # remote RM
            queries = self.tokenizer.batch_decode(sequences.cpu(), skip_special_tokens=False)
            # TODO queries format
            r = remote_rm_fn(self.remote_rm_url, queries=queries).to(device=action_log_probs.device)
        else:
            # local RM
            r = self.reward_model(sequences, attention_mask)

        reward, kl = compute_reward(
            r,
            self.kl_ctl.value,
            action_log_probs,
            base_action_log_probs,
            action_mask=action_mask,
        )
        advantage, returns = self.get_advantages_and_returns(
            value,
            reward,
            action_mask,
            generate_kwargs["gamma"],
            generate_kwargs["lambd"],
        )

        info = {
            "kl": masked_mean(kl, action_mask, dim=-1),
            "reward": r,
            "return": reward.sum(dim=-1),
            "response_length": action_mask.float().sum(dim=-1),
            "total_length": attention_mask.float().sum(dim=-1),
        }
        # reset model state
        self.actor.train()
        self.critic.train()

        return Experience(
            sequences,
            action_log_probs,
            value,
            returns,
            advantage,
            attention_mask,
            action_mask,
            info,
        )

    @torch.no_grad()
    def get_advantages_and_returns(
        self,
        values: torch.Tensor,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
        gamma: float,
        lambd: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Function that computes advantages and returns from rewards and values.
        Calculated as in the original PPO paper: https://arxiv.org/abs/1707.06347
        Note that rewards may include a KL divergence loss term.

        Advantages looks like this:
        Adv1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
              - V1 + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Returns looks like this:
        Ret1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
                   + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Input:
        - values: Tensor of shape (batch_size, response_size)
        - rewards: Tensor of shape (batch_size, response_size)

        Output:
        - advantages: Tensor of shape (batch_size, response_size)
        - returns: Tensor of shape (batch_size, response_size)
        """
        lastgaelam = 0
        advantages_reversed = []
        response_length = rewards.size(1)

        # Mask invalid responses
        values = action_mask * values
        rewards = action_mask * rewards

        for t in reversed(range(response_length)):
            nextvalues = values[:, t + 1] if t < response_length - 1 else 0.0
            delta = rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lambd * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values
        return advantages.detach(), returns


class GrpoExperienceMaker(ABC):
    """
    Naive experience maker.
    """

    def __init__(
        self,
        actor: Actor,
        reward_model: nn.Module,
        initial_model: Actor,
        tokenizer,
        prompt_max_len: int,
        kl_controller,
        strategy=None,
        remote_rm_url: str = None,
        reward_fn=None,
    ) -> None:
        super().__init__()
        self.actor = actor
        self.reward_model = reward_model
        self.remote_rm_url = remote_rm_url
        self.initial_model = initial_model
        self.tokenizer = tokenizer
        self.prompt_max_len = prompt_max_len
        self.kl_ctl = kl_controller
        self.strategy = strategy
        self.reward_fn = reward_fn

    # tokenizer
    def tokenize_fn(self, texts, max_length, device):
        batch = self.tokenizer(
            texts,
            return_tensors="pt",
            add_special_tokens=False,
            max_length=max_length,
            padding=True,
            truncation=True,
        )
        return {k: v.to(device) for k, v in batch.items()}

    @torch.no_grad()
    def make_experience(self, prompts: Union[str, List[str]], golden_answers: Union[str, List[str]], **generate_kwargs) -> Experience:
        self.actor.eval()
        self.initial_model.eval()
        if self.reward_model is not None:
            self.reward_model.eval()

        # generate seq
        inputs = self.tokenize_fn(prompts, self.prompt_max_len, device="cuda")
        sequences, attention_mask, action_mask = self.actor.generate(**inputs, **generate_kwargs)
        num_actions = action_mask.size(1)

        # log probs
        action_log_probs = self.actor(sequences, num_actions, attention_mask)

        # init log probs
        base_action_log_probs = self.initial_model(sequences, num_actions, attention_mask)

        # rewards
        test_mode = False
        if test_mode:
            r = torch.tensor([0. for _ in range(len(sequences))]).to(sequences.device)
        elif golden_answers is not None:
            queries = self.tokenizer.batch_decode(sequences.cpu(), skip_special_tokens=True)
            rewards = []
            for idx in range(len(queries)):
                match = re.findall(ANSWER_PATTERN, queries[idx][len(prompts[0]):])
                extracted_answer = match[0] if match else ""
                # print(extracted_answer, queries[idx])
                score = float(math_reward(extracted_answer, golden_answers[0]))
                rewards.append(score)
            r = torch.Tensor(rewards).to(sequences.device)
            print("reward", rewards)
        else:
            if self.remote_rm_url is not None:
                # remote RM
                queries = self.tokenizer.batch_decode(sequences.cpu(), skip_special_tokens=False)
                # raw_queries = self.tokenizer.batch_decode(sequences.cpu(), skip_special_tokens=True)
                # TODO queries format
                raw_queries = self.tokenizer.batch_decode(sequences.cpu(), skip_special_tokens=True)
                # patten_prompt = "User: ([\s\S]*)\nAssistant: "
                patten_prompt = "User: ([\s\S]*)Assistant: "
                patten_answer = 'Assistant: ([\s\S]*)'
                input_template = "User: {}Assistant: "
                queries_list = []
                for d, rd in zip(queries, raw_queries):
                    try:
                        prompt = re.findall(patten_prompt, d)[0]
                    except:
                        print(len(sequences), sequences.shape)
                        prompt = d
                    answer = rd[len(input_template.format(prompt)):]
                    queries_list.append({"user": prompt, "assistant": answer})
                r = remote_rm_fn(self.remote_rm_url, queries=queries_list).to(device=action_log_probs.device)
                # # remote RM
                # queries = self.tokenizer.batch_decode(sequences.cpu(), skip_special_tokens=False)
                # # raw_queries = self.tokenizer.batch_decode(sequences.cpu(), skip_special_tokens=True)
                # # TODO queries format
                # raw_queries = self.tokenizer.batch_decode(sequences.cpu(), skip_special_tokens=True)
                # # patten_prompt = "\\<\\|start_header_id\\|\\>user\\<\\|end_header_id\\|\\>\n\n([\s\S]*)\\<\\|eot_id\\|\\>"
                # # patten_answer = "\\<\\|start_header_id\\|\\>assistant\\<\\|end_header_id\\|\\>\n\n([\s\S]*)"
                # patten_prompt = "User: ([\s\S]*)\nAssistant: "
                # patten_answer = '\nAssistant: ([\s\S]*)'
                # input_template = "User: {}\nAssistant: "
                # queries_list = []
                # for d, rd in zip(queries, raw_queries):
                #     prompt = re.findall(patten_prompt, d)[0]
                #     answer = rd[len(input_template.format(prompt)):]
                #     # answer = re.findall(patten_answer, d)[0]
                #     queries_list.append({"user": prompt, "assistant": answer})
                #     print("\n\n--------------- prompt \n\n", prompt)
                #     print("\n\n--------------- answer \n\n", answer)
                # # # patten_prompt = "\\<\\|start_header_id\\|\\>user\\<\\|end_header_id\\|\\>\n\n([\s\S]*)\\<\\|eot_id\\|\\>\n\n\n\n\\<\\|start_header_id\\|\\>"
                # # patten_prompt = "\\<\\|start_header_id\\|\\>user\\<\\|end_header_id\\|\\>\n\n([\s\S]*)\\<\\|eot_id\\|\\>"
                # # patten_answer = "\\<\\|start_header_id\\|\\>assistant\\<\\|end_header_id\\|\\>\n\n([\s\S]*)\\<\\|eot_id\\|\\>"
                # # patten_answer_ = "\\<\\|start_header_id\\|\\>assistant\\<\\|end_header_id\\|\\>\n\n([\s\S]*)"
                # # queries_list = []
                # # for d in queries:
                # #     prompt = re.findall(patten_prompt, d)[0]
                # #     answer_res = re.findall(patten_answer, d)
                # #     if answer_res is None or len(answer_res) == 0:
                # #         answer = re.findall(patten_answer_, d)[0]
                # #     else:
                # #         answer = answer_res[0]
                # #     print(prompt, "\n\n\n\n", answer)
                # #     queries_list.append({"user": prompt, "assistant": answer})
                # r = remote_rm_fn(self.remote_rm_url, queries=queries_list).to(device=action_log_probs.device)
            else:
                # local RM
                r = self.reward_model(sequences, attention_mask)
            
        reward, kl = compute_reward(
            (r - r.mean(dim=-1, keepdim=True)) / r.std(dim=-1, keepdim=True).clamp(1e-8, None),
            self.kl_ctl.value,
            action_log_probs,
            base_action_log_probs,
            action_mask=action_mask,
        )
        value = torch.zeros_like(reward).to(reward.device)
        advantage, returns = self.get_advantages_and_returns(
            value,
            reward,
            action_mask,
            generate_kwargs["gamma"],
            generate_kwargs["lambd"],
        )

        info = {
            "kl": masked_mean(kl, action_mask, dim=-1),
            "reward": r,
            "return": reward.sum(dim=-1),
            "response_length": action_mask.float().sum(dim=-1),
            "total_length": attention_mask.float().sum(dim=-1),
        }
        # reset model state
        self.actor.train()

        return Experience(
            sequences,
            action_log_probs,
            value,
            returns,
            advantage,
            attention_mask,
            action_mask,
            info,
        )

    @torch.no_grad()
    def get_advantages_and_returns(
        self,
        values: torch.Tensor,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
        gamma: float,
        lambd: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Function that computes advantages and returns from rewards and values.
        Calculated as in the original PPO paper: https://arxiv.org/abs/1707.06347
        Note that rewards may include a KL divergence loss term.

        Advantages looks like this:
        Adv1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
              - V1 + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Returns looks like this:
        Ret1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
                   + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Input:
        - values: Tensor of shape (batch_size, response_size)
        - rewards: Tensor of shape (batch_size, response_size)

        Output:
        - advantages: Tensor of shape (batch_size, response_size)
        - returns: Tensor of shape (batch_size, response_size)
        """
        lastgaelam = 0
        advantages_reversed = []
        response_length = rewards.size(1)

        # Mask invalid responses
        values = action_mask * values
        rewards = action_mask * rewards

        for t in reversed(range(response_length)):
            nextvalues = values[:, t + 1] if t < response_length - 1 else 0.0
            delta = rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lambd * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values
        return advantages.detach(), returns


class RemaxExperienceMaker(ABC):
    """
    Naive experience maker.
    """

    def __init__(
        self,
        actor: Actor,
        reward_model: nn.Module,
        initial_model: Actor,
        tokenizer,
        prompt_max_len: int,
        kl_controller,
        strategy=None,
        remote_rm_url: str = None,
        reward_fn=None,
    ) -> None:
        super().__init__()
        self.actor = actor
        self.reward_model = reward_model
        self.remote_rm_url = remote_rm_url
        self.initial_model = initial_model
        self.tokenizer = tokenizer
        self.prompt_max_len = prompt_max_len
        self.kl_ctl = kl_controller
        self.strategy = strategy
        self.reward_fn = reward_fn

    # tokenizer
    def tokenize_fn(self, texts, max_length, device):
        batch = self.tokenizer(
            texts,
            return_tensors="pt",
            add_special_tokens=False,
            max_length=max_length,
            padding=True,
            truncation=True,
        )
        return {k: v.to(device) for k, v in batch.items()}

    @torch.no_grad()
    def make_experience(self, prompts: Union[str, List[str]], **generate_kwargs) -> Experience:
        self.actor.eval()
        self.initial_model.eval()
        if self.reward_model is not None:
            self.reward_model.eval()

        # generate seq
        inputs = self.tokenize_fn(prompts, self.prompt_max_len, device="cuda")
        sequences, attention_mask, action_mask = self.actor.generate(**inputs, **generate_kwargs)
        num_actions = action_mask.size(1)

        # log probs
        action_log_probs = self.actor(sequences, num_actions, attention_mask)
        

        # init log probs
        base_action_log_probs = self.initial_model(sequences, num_actions, attention_mask)

        # TODO baseline value
        # values
        greedy_args = {
            "top_k": None, 
            "top_p": None, 
            "do_sample": False,
        }
        for key in generate_kwargs.keys():
            if key not in ['top_k', 'top_p', 'do_sample']:
                greedy_args[key] = generate_kwargs[key]
        assert greedy_args['do_sample'] == False
        base_sequences, base_attention_mask, base_action_mask = self.actor.generate(**inputs, **greedy_args)

        # rewards
        if self.remote_rm_url is not None:
            # remote RM
            queries = self.tokenizer.batch_decode(sequences.cpu(), skip_special_tokens=False)
            r = remote_rm_fn(self.remote_rm_url, queries=queries).to(device=action_log_probs.device)
            base_queries = self.tokenizer.batch_decode(base_sequences.cpu(), skip_special_tokens=False)
            value = remote_rm_fn(self.remote_rm_url, queries=base_queries).to(device=action_log_probs.device)
        else:
            # local RM
            r = self.reward_model(sequences, attention_mask)
            value = self.reward_model(base_sequences, base_attention_mask)

        reward, kl = compute_reward_remax(
            r,
            value,
            self.kl_ctl.value,
            action_log_probs,
            base_action_log_probs,
            action_mask=action_mask,
            base_action_mask=base_action_mask,
        )
        value = torch.zeros_like(reward).to(value.device)
        advantage, returns = self.get_advantages_and_returns(
            value,
            reward,
            action_mask,
            generate_kwargs["gamma"],
            generate_kwargs["lambd"],
        )

        info = {
            "kl": masked_mean(kl, action_mask, dim=-1),
            "reward": r,
            "return": reward.sum(dim=-1),
            "response_length": action_mask.float().sum(dim=-1),
            "total_length": attention_mask.float().sum(dim=-1),
        }
        # reset model state
        self.actor.train()

        return Experience(
            sequences,
            action_log_probs,
            value,
            returns,
            advantage,
            attention_mask,
            action_mask,
            info,
        )

    @torch.no_grad()
    def get_advantages_and_returns(
        self,
        values: torch.Tensor,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
        gamma: float,
        lambd: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Function that computes advantages and returns from rewards and values.
        Calculated as in the original PPO paper: https://arxiv.org/abs/1707.06347
        Note that rewards may include a KL divergence loss term.

        Advantages looks like this:
        Adv1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
              - V1 + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Returns looks like this:
        Ret1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
                   + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Input:
        - values: Tensor of shape (batch_size, response_size)
        - rewards: Tensor of shape (batch_size, response_size)

        Output:
        - advantages: Tensor of shape (batch_size, response_size)
        - returns: Tensor of shape (batch_size, response_size)
        """
        lastgaelam = 0
        advantages_reversed = []
        response_length = rewards.size(1)

        # Mask invalid responses
        values = action_mask * values
        rewards = action_mask * rewards

        for t in reversed(range(response_length)):
            nextvalues = values[:, t + 1] if t < response_length - 1 else 0.0
            delta = rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lambd * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values
        return advantages.detach(), returns



class RemoteExperienceMaker(NaiveExperienceMaker):
    def __init__(self, *args, vllm_engines: List = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.vllm_engines = vllm_engines

    @torch.no_grad()
    def make_experience(self, prompts: Union[str, List[str]], **generate_kwargs) -> Experience:
        self.actor.eval()
        device = torch.cuda.current_device()

        # generate sequence
        start = time.time()
        sequences, attention_mask, action_mask = (
            self._generate_local(prompts, **generate_kwargs)
            if self.vllm_engines is None
            else self._generate_vllm(prompts, **generate_kwargs)
        )
        generate_time = time.time() - start

        num_actions = action_mask.size(1)
        sequences_cpu, attention_mask_cpu = (
            sequences.to("cpu"),
            attention_mask.to("cpu"),
        )

        # init log probs
        base_action_log_probs_ref = self.initial_model.forward.remote(sequences_cpu, num_actions, attention_mask_cpu)

        # values
        value_ref = self.critic.forward.remote(sequences_cpu, num_actions, attention_mask_cpu)

        # avoid CUDA OOM when colocate models
        if self.strategy.args.colocate_critic_reward:
            ray.get([value_ref])
            ray.get([self.critic.empty_cache.remote()])

        if self.strategy.args.colocate_actor_ref:
            ray.get([base_action_log_probs_ref])
            ray.get([self.initial_model.empty_cache.remote()])

        # rewards
        r_refs = []
        # support remote RM API with ray
        if not self.remote_rm_url:
            for rm in self.reward_model:
                r_refs.append(rm.forward.remote(sequences_cpu, attention_mask_cpu))
        else:
            # remote RM
            for rm in self.remote_rm_url:
                queries = self.tokenizer.batch_decode(sequences.cpu(), skip_special_tokens=False)
                r = remote_rm_fn_ray.remote(rm, queries=queries)
                r_refs.append(r)

        # log probs
        start = time.time()
        action_log_probs = self.actor(sequences, num_actions, attention_mask)
        actor_time = time.time() - start

        # wait initial/critic/reward model done
        start = time.time()
        ref_values = ray.get([base_action_log_probs_ref, value_ref] + r_refs)
        wait_time = time.time() - start

        base_action_log_probs, value, rewards = ref_values[0], ref_values[1], ref_values[2:]
        base_action_log_probs, value = base_action_log_probs.to(device), value.to(device)
        rewards = [r.to(device) for r in rewards]
        r = self.reward_fn(rewards) if len(rewards) > 0 else rewards[0]

        # avoid CUDA OOM when colocate models
        if self.strategy.args.colocate_critic_reward and not self.remote_rm_url:
            ray.get([self.reward_model[0].empty_cache.remote()])

        if self.strategy.args.colocate_actor_ref:
            torch.cuda.empty_cache()

        reward, kl = compute_reward(
            r,
            self.kl_ctl.value,
            action_log_probs,
            base_action_log_probs,
            action_mask=action_mask,
        )
        advantage, returns = self.get_advantages_and_returns(
            value,
            reward,
            action_mask,
            generate_kwargs["gamma"],
            generate_kwargs["lambd"],
        )

        info = {
            "kl": masked_mean(kl, action_mask, dim=-1),
            "reward": r,
            "return": reward.sum(dim=-1),
            "response_length": action_mask.float().sum(dim=-1),
            "total_length": attention_mask.float().sum(dim=-1),
        }

        if self.strategy.args.perf:
            batch_size = 1 if isinstance(prompts, str) else len(prompts)
            info["generate_time"] = torch.full((batch_size,), generate_time, device=device)
            info["actor_time"] = torch.full((batch_size,), actor_time, device=device)
            info["wait_time"] = torch.full((batch_size,), wait_time, device=device)

        experience = Experience(
            sequences,
            action_log_probs,
            value,
            returns,
            advantage,
            attention_mask,
            action_mask,
            info,
        )

        # send experience to critic
        experience_cpu = deepcopy(experience)
        experience_cpu.to_device("cpu")
        self._ref = self.critic.append.remote(experience_cpu)

        self.actor.train()  # reset model state
        return experience

    def _generate_local(self, prompts: List[str], **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs = self.tokenize_fn(prompts, self.prompt_max_len, device="cuda")
        return self.actor.generate(**inputs, **kwargs)

    def _generate_vllm(self, prompts: List[str], **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from vllm import SamplingParams

        # round-robin load balance
        rank = torch.distributed.get_rank()
        llm = self.vllm_engines[rank % len(self.vllm_engines)]

        sampling_params = SamplingParams(
            temperature=kwargs.get("temperature", 1.0),
            top_p=kwargs.get("top_p", 1.0),
            top_k=kwargs.get("top_k", -1),
            max_tokens=kwargs.get("max_new_tokens", 1024),
            min_tokens=kwargs.get("min_new_tokens", 1),
            skip_special_tokens=kwargs.get("skip_special_tokens", False),
        )

        # TODO: can't pass `max_length` to vLLM's tokenizer for input truncation, remove this once it is supported.
        input_ids = self.tokenize_fn(prompts, self.prompt_max_len, device="cpu")["input_ids"]
        assert self.tokenizer.padding_side == "left", f"tokenizer padding_size should be left"
        pad_indices = (input_ids != self.tokenizer.pad_token_id).to(dtype=torch.int).argmax(dim=-1)
        prompt_token_ids = []
        for i, pad_index in enumerate(pad_indices.numpy()):
            prompt_token_ids.append(input_ids[i][pad_index:].tolist())
        outputs = ray.get(llm.generate.remote(sampling_params=sampling_params, prompt_token_ids=prompt_token_ids))

        # NOTE: concat all outputs to following format:
        #
        # | [PAD] [PAD] token token token | token token [EOS] [PAD] |
        # | token token token token token | token token [EOS] [PAD] |
        # | [PAD] [PAD] [PAD] token token | token token token [EOS] |
        # |<---------- prompt ----------->|<-------- answer ------->|
        max_input_len, max_output_len = 0, 0
        for output in outputs:
            max_input_len = max(max_input_len, len(output.prompt_token_ids))
            max_output_len = max(max_output_len, len(output.outputs[0].token_ids))

        pad_token_id, eos_token_id = self.tokenizer.pad_token_id, self.tokenizer.eos_token_id
        sequences = []
        for output in outputs:
            # left padding input
            input_len = len(output.prompt_token_ids)
            input_ids = [pad_token_id] * (max_input_len - input_len) + list(output.prompt_token_ids)

            # right padding output
            output_len = len(output.outputs[0].token_ids)
            output_ids = list(output.outputs[0].token_ids) + [pad_token_id] * (max_output_len - output_len)

            if output_ids[output_len - 1] != eos_token_id:
                output_ids[min(output_len, len(output_ids) - 1)] = eos_token_id

            # concat input and output
            sequences.append(input_ids + output_ids)

        sequences = torch.tensor(sequences)
        sequences, attention_mask, action_mask = self.actor.process_sequences(
            sequences, max_input_len, eos_token_id, pad_token_id
        )
        return sequences.to("cuda"), attention_mask.to("cuda"), action_mask.to("cuda")

    def flush(self):
        "Ensure all experience has been send to critic"
        ray.get(self._ref)
        self._ref = None


class IntentionExperienceMaker(ABC):
    """
    Naive experience maker.
    """

    def __init__(
        self,
        actor: Actor,
        critic: nn.Module,
        reward_model: nn.Module,
        initial_model: Actor,
        tokenizer,
        prompt_max_len: int,
        kl_controller,
        strategy=None,
        remote_rm_url: str = None,
        reward_fn=None,
    ) -> None:
        super().__init__()
        self.actor = actor
        self.critic = critic
        self.reward_model = reward_model
        self.remote_rm_url = remote_rm_url
        self.initial_model = initial_model
        self.tokenizer = tokenizer
        self.prompt_max_len = prompt_max_len
        self.kl_ctl = kl_controller
        self.strategy = strategy
        self.reward_fn = reward_fn

    # tokenizer
    def tokenize_fn(self, texts, max_length, device):
        batch = self.tokenizer(
            texts,
            return_tensors="pt",
            add_special_tokens=False,
            max_length=max_length,
            padding=True,
            truncation=True,
        )
        return {k: v.to(device) for k, v in batch.items()}

    @torch.no_grad()
    def make_experience(self, prompts: Union[str, List[str]], **generate_kwargs) -> ExperienceWithAction:
        self.actor.eval()
        self.critic.eval()
        self.initial_model.eval()
        if self.reward_model is not None:
            self.reward_model.eval()

        # generate seq
        inputs = self.tokenize_fn(prompts, self.prompt_max_len, device="cuda")
        # TODO action sequence
        sequences, action_sequences, attention_mask, action_mask = self.actor.generate(**inputs, **generate_kwargs)
        ############################################
        num_actions = action_mask.size(1)

        policy_embeds = None # self.actor(sequences=sequences, attention_mask=attention_mask, return_output=True, emb_mode=True)
        # log probs
        action_log_probs = self.actor(sequences, num_actions, attention_mask, act_mode=True, action_sequences=action_sequences, policy_embeds=policy_embeds)
        # action_log_probs = torch.log(F.softmax(action_logits, dim=-1))

        # init log probs
        base_action_log_probs = self.initial_model(sequences, num_actions, attention_mask, act_mode=True, action_sequences=action_sequences, policy_embeds=policy_embeds)
        # base_action_log_probs = torch.log(F.softmax(base_action_logits, dim=-1))

        # values
        value = self.critic(sequences, num_actions, attention_mask)

        # rewards
        if self.remote_rm_url is not None:
            # remote RM
            queries = self.tokenizer.batch_decode(sequences.cpu(), skip_special_tokens=False)
            r = remote_rm_fn(self.remote_rm_url, queries=queries).to(device=action_log_probs.device)
        else:
            # local RM
            r = self.reward_model(sequences, attention_mask)

        reward, kl = compute_reward(
            r,
            self.kl_ctl.value,
            action_log_probs,
            base_action_log_probs,
            action_mask=action_mask,
        )
        advantage, returns = self.get_advantages_and_returns(
            value,
            reward,
            action_mask,
            generate_kwargs["gamma"],
            generate_kwargs["lambd"],
        )

        info = {
            "kl": masked_mean(kl, action_mask, dim=-1),
            "reward": r,
            "return": reward.sum(dim=-1),
            "response_length": action_mask.float().sum(dim=-1),
            "total_length": attention_mask.float().sum(dim=-1),
        }
        # reset model state
        self.actor.train()
        self.critic.train()

        return ExperienceWithAction(
            sequences,
            action_sequences,
            # policy_embeds,
            action_log_probs,
            value,
            returns,
            advantage,
            attention_mask,
            action_mask,
            info,
        )

    @torch.no_grad()
    def get_advantages_and_returns(
        self,
        values: torch.Tensor,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
        gamma: float,
        lambd: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Function that computes advantages and returns from rewards and values.
        Calculated as in the original PPO paper: https://arxiv.org/abs/1707.06347
        Note that rewards may include a KL divergence loss term.

        Advantages looks like this:
        Adv1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
              - V1 + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Returns looks like this:
        Ret1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
                   + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Input:
        - values: Tensor of shape (batch_size, response_size)
        - rewards: Tensor of shape (batch_size, response_size)

        Output:
        - advantages: Tensor of shape (batch_size, response_size)
        - returns: Tensor of shape (batch_size, response_size)
        """
        lastgaelam = 0
        advantages_reversed = []
        response_length = rewards.size(1)

        # Mask invalid responses
        values = action_mask * values
        rewards = action_mask * rewards

        for t in reversed(range(response_length)):
            nextvalues = values[:, t + 1] if t < response_length - 1 else 0.0
            delta = rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lambd * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values
        return advantages.detach(), returns
    

class IntentionRemaxExperienceMaker(ABC):
    """
    Naive experience maker.
    """

    def __init__(
        self,
        actor: Actor,
        reward_model: nn.Module,
        initial_model: Actor,
        tokenizer,
        prompt_max_len: int,
        kl_controller,
        strategy=None,
        remote_rm_url: str = None,
        reward_fn=None,
    ) -> None:
        super().__init__()
        self.actor = actor
        self.reward_model = reward_model
        self.remote_rm_url = remote_rm_url
        self.initial_model = initial_model
        self.tokenizer = tokenizer
        self.prompt_max_len = prompt_max_len
        self.kl_ctl = kl_controller
        self.strategy = strategy
        self.reward_fn = reward_fn

    # tokenizer
    def tokenize_fn(self, texts, max_length, device):
        batch = self.tokenizer(
            texts,
            return_tensors="pt",
            add_special_tokens=False,
            max_length=max_length,
            padding=True,
            truncation=True,
        )
        return {k: v.to(device) for k, v in batch.items()}

    @torch.no_grad()
    def make_experience(self, prompts: Union[str, List[str]], **generate_kwargs) -> ExperienceWithAction:
        self.actor.eval()
        self.initial_model.eval()
        if self.reward_model is not None:
            self.reward_model.eval()

        # generate seq
        inputs = self.tokenize_fn(prompts, self.prompt_max_len, device="cuda")
        # TODO action sequence
        sequences, action_sequences, attention_mask, action_mask = self.actor.generate(**inputs, **generate_kwargs)
        ############################################
        num_actions = action_mask.size(1)

        policy_embeds = None # self.actor(sequences=sequences, attention_mask=attention_mask, return_output=True, emb_mode=True)
        # log probs
        action_log_probs = self.actor(sequences, num_actions, attention_mask, act_mode=True, action_sequences=action_sequences, policy_embeds=policy_embeds)
        # action_log_probs = torch.log(F.softmax(action_logits, dim=-1))

        # init log probs
        base_action_log_probs = self.initial_model(sequences, num_actions, attention_mask, act_mode=True, action_sequences=action_sequences, policy_embeds=policy_embeds)
        # base_action_log_probs = torch.log(F.softmax(base_action_logits, dim=-1))

        # TODO baseline value
        # values
        greedy_args = {
            "top_k": None, 
            "top_p": None, 
            "do_sample": False,
        }
        for key in generate_kwargs.keys():
            if key not in ['top_k', 'top_p', 'do_sample']:
                greedy_args[key] = generate_kwargs[key]
        assert greedy_args['do_sample'] == False
        base_sequences, _, base_attention_mask, base_action_mask = self.actor.generate(**inputs, **greedy_args, deterministic=True)

        # rewards
        if self.remote_rm_url is not None:
            # remote RM
            # queries = self.tokenizer.batch_decode(sequences.cpu(), skip_special_tokens=False)
            # r = remote_rm_fn(self.remote_rm_url, queries=queries).to(device=action_log_probs.device)
            # base_queries = self.tokenizer.batch_decode(base_sequences.cpu(), skip_special_tokens=False)
            # value = remote_rm_fn(self.remote_rm_url, queries=base_queries).to(device=action_log_probs.device)
            # remote RM
            # patten_prompt = "User: ([\s\S]*)\nAssistant: "
            patten_prompt = "User: ([\s\S]*)Assistant: "
            patten_answer = 'Assistant: ([\s\S]*)'
            input_template = "User: {}Assistant: "
            queries = self.tokenizer.batch_decode(sequences.cpu(), skip_special_tokens=False)
            # raw_queries = self.tokenizer.batch_decode(sequences.cpu(), skip_special_tokens=True)
            # TODO queries format
            raw_queries = self.tokenizer.batch_decode(sequences.cpu(), skip_special_tokens=True)
            queries_list = []
            for d, rd in zip(queries, raw_queries):
                try:
                    prompt = re.findall(patten_prompt, d)[0]
                except:
                    prompt = d
                answer = rd[len(input_template.format(prompt)):]
                queries_list.append({"user": prompt, "assistant": answer})
            r = remote_rm_fn(self.remote_rm_url, queries=queries_list).to(device=action_log_probs.device)

            queries_base = self.tokenizer.batch_decode(base_sequences.cpu(), skip_special_tokens=False)
            # raw_queries = self.tokenizer.batch_decode(sequences.cpu(), skip_special_tokens=True)
            # TODO queries format
            raw_queries_base = self.tokenizer.batch_decode(base_sequences.cpu(), skip_special_tokens=True)
            queries_list_base = []
            for db, rdb in zip(queries_base, raw_queries_base):
                try:
                    prompt_base = re.findall(patten_prompt, db)[0]
                except:
                    prompt_base = db
                answer_base = rdb[len(input_template.format(prompt_base)):]
                queries_list_base.append({"user": prompt_base, "assistant": answer_base})
            value = remote_rm_fn(self.remote_rm_url, queries=queries_list_base).to(device=action_log_probs.device)
            # print(r - value)
        else:
            # local RM
            r = self.reward_model(sequences, attention_mask)
            value = self.reward_model(base_sequences, base_attention_mask)

        reward, kl = compute_reward_remax(
            r,
            value,
            self.kl_ctl.value,
            action_log_probs,
            base_action_log_probs,
            action_mask=action_mask,
            base_action_mask=base_action_mask,
        )
        value = torch.zeros_like(reward).to(value.device)
        advantage, returns = self.get_advantages_and_returns(
            value,
            reward,
            action_mask,
            generate_kwargs["gamma"],
            generate_kwargs["lambd"],
        )

        info = {
            "kl": masked_mean(kl, action_mask, dim=-1),
            "reward": r,
            "return": reward.sum(dim=-1),
            "response_length": action_mask.float().sum(dim=-1),
            "total_length": attention_mask.float().sum(dim=-1),
        }
        # reset model state
        self.actor.train()

        return ExperienceWithAction(
            sequences,
            action_sequences,
            # policy_embeds,
            action_log_probs,
            value,
            returns,
            advantage,
            attention_mask,
            action_mask,
            info,
        )

    @torch.no_grad()
    def get_advantages_and_returns(
        self,
        values: torch.Tensor,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
        gamma: float,
        lambd: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Function that computes advantages and returns from rewards and values.
        Calculated as in the original PPO paper: https://arxiv.org/abs/1707.06347
        Note that rewards may include a KL divergence loss term.

        Advantages looks like this:
        Adv1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
              - V1 + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Returns looks like this:
        Ret1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
                   + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Input:
        - values: Tensor of shape (batch_size, response_size)
        - rewards: Tensor of shape (batch_size, response_size)

        Output:
        - advantages: Tensor of shape (batch_size, response_size)
        - returns: Tensor of shape (batch_size, response_size)
        """
        lastgaelam = 0
        advantages_reversed = []
        response_length = rewards.size(1)

        # Mask invalid responses
        values = action_mask * values
        rewards = action_mask * rewards

        for t in reversed(range(response_length)):
            nextvalues = values[:, t + 1] if t < response_length - 1 else 0.0
            delta = rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lambd * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values
        return advantages.detach(), returns
    

class IntentionGrpoExperienceMaker(ABC):
    """
    Naive experience maker.
    """

    def __init__(
        self,
        actor: Actor,
        reward_model: nn.Module,
        initial_model: Actor,
        tokenizer,
        prompt_max_len: int,
        kl_controller,
        strategy=None,
        remote_rm_url: str = None,
        reward_fn=None,
    ) -> None:
        super().__init__()
        self.actor = actor
        self.reward_model = reward_model
        self.remote_rm_url = remote_rm_url
        self.initial_model = initial_model
        self.tokenizer = tokenizer
        self.prompt_max_len = prompt_max_len
        self.kl_ctl = kl_controller
        self.strategy = strategy
        self.reward_fn = reward_fn

    # tokenizer
    def tokenize_fn(self, texts, max_length, device):
        batch = self.tokenizer(
            texts,
            return_tensors="pt",
            add_special_tokens=False,
            max_length=max_length,
            padding=True,
            truncation=True,
        )
        return {k: v.to(device) for k, v in batch.items()}

    @torch.no_grad()
    def make_experience(self, prompts: Union[str, List[str]], golden_answers: Union[str, List[str]], **generate_kwargs) -> ExperienceWithAction:
        self.actor.eval()
        self.initial_model.eval()
        if self.reward_model is not None:
            self.reward_model.eval()

        # generate seq
        inputs = self.tokenize_fn(prompts, self.prompt_max_len, device="cuda")
        # TODO action sequence
        sequences, action_sequences, attention_mask, action_mask = self.actor.generate(**inputs, **generate_kwargs)
        ############################################
        num_actions = action_mask.size(1)

        policy_embeds = None # self.actor(sequences=sequences, attention_mask=attention_mask, return_output=True, emb_mode=True)
        # log probs
        action_log_probs = self.actor(sequences, num_actions, attention_mask, act_mode=True, action_sequences=action_sequences, policy_embeds=policy_embeds)
        # action_log_probs = torch.log(F.softmax(action_logits, dim=-1))

        # init log probs
        base_action_log_probs = self.initial_model(sequences, num_actions, attention_mask, act_mode=True, action_sequences=action_sequences, policy_embeds=policy_embeds)
        # base_action_log_probs = torch.log(F.softmax(base_action_logits, dim=-1))

        # rewards
        test_mode = False
        if test_mode:
            r = torch.Tensor([0. for i in range(len(sequences))]).to(sequences.device)
        elif golden_answers is not None:
            queries = self.tokenizer.batch_decode(sequences.cpu(), skip_special_tokens=True)
            rewards = []
            for idx in range(len(queries)):
                match = re.findall(ANSWER_PATTERN, queries[idx][len(prompts[0]):])
                extracted_answer = match[0] if match else ""
                # print(extracted_answer, queries[idx])
                score = float(math_reward(extracted_answer, golden_answers[0]))
                rewards.append(score)
            r = torch.Tensor(rewards).to(sequences.device)
            print("reward", rewards)
        else:
            if self.remote_rm_url is not None:
                # remote RM
                queries = self.tokenizer.batch_decode(sequences.cpu(), skip_special_tokens=False)
                # raw_queries = self.tokenizer.batch_decode(sequences.cpu(), skip_special_tokens=True)
                # TODO queries format
                raw_queries = self.tokenizer.batch_decode(sequences.cpu(), skip_special_tokens=True)
                # patten_prompt = "User: ([\s\S]*)\nAssistant: "
                patten_prompt = "User: ([\s\S]*)Assistant: "
                patten_answer = 'Assistant: ([\s\S]*)'
                input_template = "User: {}Assistant: "
                queries_list = []
                for d, rd in zip(queries, raw_queries):
                    try:
                        prompt = re.findall(patten_prompt, d)[0]
                    except:
                        prompt = d
                    answer = rd[len(input_template.format(prompt)):]
                    queries_list.append({"user": prompt, "assistant": answer})
                r = remote_rm_fn(self.remote_rm_url, queries=queries_list).to(device=action_log_probs.device)
            else:
                # local RM
                r = self.reward_model(sequences, attention_mask)

        reward, kl = compute_reward(
            (r - r.mean(dim=-1, keepdim=True)) / r.std(dim=-1, keepdim=True).clamp(1e-8, None),
            self.kl_ctl.value,
            action_log_probs,
            base_action_log_probs,
            action_mask=action_mask,
        )
        value = torch.zeros_like(reward).to(reward.device)
        advantage, returns = self.get_advantages_and_returns(
            value,
            reward,
            action_mask,
            generate_kwargs["gamma"],
            generate_kwargs["lambd"],
        )

        info = {
            "kl": masked_mean(kl, action_mask, dim=-1),
            "reward": r,
            "return": reward.sum(dim=-1),
            "response_length": action_mask.float().sum(dim=-1),
            "total_length": attention_mask.float().sum(dim=-1),
        }
        # reset model state
        self.actor.train()

        return ExperienceWithAction(
            sequences,
            action_sequences,
            # policy_embeds,
            action_log_probs,
            value,
            returns,
            advantage,
            attention_mask,
            action_mask,
            info,
        )

    @torch.no_grad()
    def get_advantages_and_returns(
        self,
        values: torch.Tensor,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
        gamma: float,
        lambd: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Function that computes advantages and returns from rewards and values.
        Calculated as in the original PPO paper: https://arxiv.org/abs/1707.06347
        Note that rewards may include a KL divergence loss term.

        Advantages looks like this:
        Adv1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
              - V1 + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Returns looks like this:
        Ret1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
                   + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Input:
        - values: Tensor of shape (batch_size, response_size)
        - rewards: Tensor of shape (batch_size, response_size)

        Output:
        - advantages: Tensor of shape (batch_size, response_size)
        - returns: Tensor of shape (batch_size, response_size)
        """
        lastgaelam = 0
        advantages_reversed = []
        response_length = rewards.size(1)

        # Mask invalid responses
        values = action_mask * values
        rewards = action_mask * rewards

        for t in reversed(range(response_length)):
            nextvalues = values[:, t + 1] if t < response_length - 1 else 0.0
            delta = rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lambd * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values
        return advantages.detach(), returns
    


class InteExperienceMaker(ABC):
    """
    Naive experience maker.
    """

    def __init__(
        self,
        actor: Actor,
        reward_model: nn.Module,
        tokenizer,
        prompt_max_len: int,
        kl_controller,
        strategy=None,
        remote_rm_url: str = None,
        reward_fn=None,
        eval_mode=False,
    ) -> None:
        super().__init__()
        self.actor = actor
        self.reward_model = reward_model
        self.remote_rm_url = remote_rm_url
        self.tokenizer = tokenizer
        self.prompt_max_len = prompt_max_len
        self.kl_ctl = kl_controller
        self.strategy = strategy
        self.reward_fn = reward_fn

        self.eval_mode = eval_mode

    # tokenizer
    def tokenize_fn(self, texts, max_length, device):
        batch = self.tokenizer(
            texts,
            return_tensors="pt",
            add_special_tokens=False,
            max_length=max_length,
            padding=True,
            truncation=True,
        )
        return {k: v.to(device) for k, v in batch.items()}

    @torch.no_grad()
    def make_experience(self, prompts: Union[str, List[str]], **generate_kwargs):
        self.actor.eval()
        if self.reward_model is not None:
            self.reward_model.eval()

        # generate seq
        inputs = self.tokenize_fn(prompts, self.prompt_max_len, device="cuda")
        deterministic = False if not self.eval_mode else True
        sequences, action_sequences, attention_mask, action_mask = self.actor.generate(**inputs, **generate_kwargs, deterministic=deterministic)
        num_actions = action_mask.size(1)

        # log probs
        action_log_probs = self.actor(sequences, num_actions, attention_mask)

        # rewards
        if self.remote_rm_url is not None:
            # remote RM
            queries = self.tokenizer.batch_decode(sequences.cpu(), skip_special_tokens=False)
            r = remote_rm_fn(self.remote_rm_url, queries=queries).to(device=action_log_probs.device)
        else:
            # local RM
            r = self.reward_model(sequences, attention_mask)

        info = {
            "reward": r,
            "response_length": action_mask.float().sum(dim=-1),
            "total_length": attention_mask.float().sum(dim=-1),
        }

        return info
    


class AnswerMaker(ABC):
    """
    Naive experience maker.
    """

    def __init__(
        self,
        actor: Actor,
        tokenizer,
        prompt_max_len: int,
        kl_controller,
        strategy=None,
        remote_rm_url: str = None,
        reward_fn=None,
    ) -> None:
        super().__init__()
        self.actor = actor
        self.remote_rm_url = remote_rm_url
        self.tokenizer = tokenizer
        self.prompt_max_len = prompt_max_len
        self.kl_ctl = kl_controller
        self.strategy = strategy
        self.reward_fn = reward_fn

    # tokenizer
    def tokenize_fn(self, texts, max_length, device):
        batch = self.tokenizer(
            texts,
            return_tensors="pt",
            add_special_tokens=False,
            max_length=max_length,
            padding=True,
            truncation=True,
        )
        return {k: v.to(device) for k, v in batch.items()}

    @torch.no_grad()
    def make_experience(self, prompts: Union[str, List[str]], **generate_kwargs) -> Experience:
        self.actor.eval()

        # generate seq
        inputs = self.tokenize_fn(prompts, self.prompt_max_len, device="cuda")
        prompt_length = inputs["input_ids"].shape[1]
        sequences, attention_mask, action_mask = self.actor.generate(**inputs, **generate_kwargs)
        num_actions = action_mask.size(1)
        print(attention_mask.float().sum(1))
        print(attention_mask.float().sum(1).mean())

        return sequences[:, prompt_length:]
    

class IntentionAnswerMaker(ABC):
    """
    Naive experience maker.
    """

    def __init__(
        self,
        actor: Actor,
        tokenizer,
        prompt_max_len: int,
        kl_controller,
        strategy=None,
        remote_rm_url: str = None,
        reward_fn=None,
    ) -> None:
        super().__init__()
        self.actor = actor
        self.remote_rm_url = remote_rm_url
        self.tokenizer = tokenizer
        self.prompt_max_len = prompt_max_len
        self.kl_ctl = kl_controller
        self.strategy = strategy
        self.reward_fn = reward_fn

    # tokenizer
    def tokenize_fn(self, texts, max_length, device):
        batch = self.tokenizer(
            texts,
            return_tensors="pt",
            add_special_tokens=False,
            max_length=max_length,
            padding=True,
            truncation=True,
        )
        return {k: v.to(device) for k, v in batch.items()}

    @torch.no_grad()
    def make_experience(self, prompts: Union[str, List[str]], **generate_kwargs) -> ExperienceWithAction:
        self.actor.eval()

        # generate seq
        inputs = self.tokenize_fn(prompts, self.prompt_max_len, device="cuda")
        prompt_length = inputs["input_ids"].shape[1]
        # TODO action sequence
        greedy_args = {
            "top_k": None, 
            "top_p": None, 
            "do_sample": False,
        }
        for key in generate_kwargs.keys():
            if key not in ['top_k', 'top_p', 'do_sample']:
                greedy_args[key] = generate_kwargs[key]
        assert greedy_args['do_sample'] == False
        # print(greedy_args)
        sequences, action_sequences, attention_mask, action_mask = self.actor.generate(**inputs, **greedy_args, deterministic=False)
        # sequences, action_sequences, attention_mask, action_mask = self.actor.generate(**inputs, **generate_kwargs)
        ############################################
        num_actions = action_mask.size(1)
        print(attention_mask.float().sum(1))
        print(attention_mask.float().sum(1).mean())
        
        return sequences[:, prompt_length:]