from torch.utils.data import Dataset
from tqdm import tqdm
from .utils import exist_and_not_none
import re


def preprocess_data(data, input_template=None, input_key="input", apply_chat_template=None) -> str:
    if apply_chat_template:
        format_prompt = [{
            "role": "user",
            "content": data[input_key],
        }]
        prompt = apply_chat_template(format_prompt, tokenize=False, add_generation_prompt=True)
    else:
        prompt = "Please reason step by step, and put your final answer within \\boxed{}.\n\n" + data[input_key]
        if input_template:
            prompt = input_template.format(prompt)
    return prompt


# def preprocess_data(data, input_template=None, input_key="input", apply_chat_template=None) -> str:
#     if apply_chat_template:
#         prompt = apply_chat_template(data[input_key], tokenize=False, add_generation_prompt=True)
#     else:
#         prompt = data[input_key]
#         if input_template:
#             prompt = input_template.format(prompt)
#     return prompt

def preprocess_data_split(data, input_template=None, input_key="input", apply_chat_template=None) -> str:
    patten_prompt = '## Human([\s\S]*)## Assistant:\n'
    patten_answer = '## Assistant:\n([\s\S]*)'
    fill_prompt = '## Human{}## Assistant:\n'

    pos_sentence, neg_sentence = data['text']
    prompt = re.findall(patten_prompt, pos_sentence)[0]
    prompt = fill_prompt.format(prompt)
    return prompt


class PromptDataset(Dataset):
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
        for data in tqdm(dataset, desc="Preprocessing data", disable=not self.strategy.is_rank_0()):
            prompt = preprocess_data(data, input_template, input_key, apply_chat_template)
            self.prompts.append(prompt)

    def __len__(self):
        length = len(self.prompts)
        return length * self.n_samples_per_prompt

    def __getitem__(self, idx):
        return self.prompts[idx // self.n_samples_per_prompt]


class PromptMergeDataset(Dataset):
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
        for data in tqdm(dataset, desc="Preprocessing data", disable=not self.strategy.is_rank_0()):
            prompt = preprocess_data_split(data, input_template, input_key, apply_chat_template)
            self.prompts.append(prompt)

    def __len__(self):
        length = len(self.prompts)
        return length * self.n_samples_per_prompt

    def __getitem__(self, idx):
        return self.prompts[idx // self.n_samples_per_prompt]