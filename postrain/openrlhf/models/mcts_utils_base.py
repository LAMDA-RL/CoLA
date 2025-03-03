import numpy as np
import torch
import torch.nn.functional as F
from collections import deque
from openrlhf.utils.remote_rm_utils import remote_rm_fn, remote_rm_fn_ray


additional_length = 128
max_length = 1024
input_template = "User: {}\nAssistant: " 
def is_terminal(action, eos_id):
    return eos_id in action

class MCTS:
    def __init__(self, model, tokenizer, cpuct=0.7, gamma=0.9, b1=5, b2=4, T=32, remote_rm_url=None):
        self.model = model
        self.tokenizer = tokenizer
        self.cpuct = cpuct
        self.gamma = gamma
        self.b1 = b1
        self.b2 = b2
        self.T = T
        self.leaves = []
        self.remote_rm_url = remote_rm_url
        
    def search(self, prompt, golden_answer):
        root = Node(prompt=prompt, state="", action="", golden_answer=golden_answer, terminal=False, model=self.model, tokenizer=self.tokenizer, parent=None, b1=self.b1, b2=self.b2, policy_prob=0.5, remote_rm_url=self.remote_rm_url, action_sequence=[])
        for round in range(self.T):
            # print("round", round)
            node = root
            while not node.terminal:
                if (not node.terminal) and (len(node.children.items()) < node.expand_count):
                    node.expand()
                    break
                selected_node = node.select(self.cpuct)
                if selected_node is None:
                    break
                node = selected_node
        return root.get_response()
    

class Node:
    def __init__(self, prompt, state, action, golden_answer, terminal, model, tokenizer, parent=None, b1=5, b2=4, policy_prob=0.5, remote_rm_url=None, action_sequence=[]):
        # self.id = time.time()
        self.prompt = prompt
        self.state = state
        self.action = action
        self.max_depth = 4
        self.terminal = terminal
        self.golden_answer = golden_answer
        self.model = model
        self.tokenizer = tokenizer
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.value = 0
        self.q_values = {}
        self.depth = parent.depth + 1 if parent else 0
        self.R = 0
        self.b1 = b1
        self.correct = 0
        self.b2 = b2
        self.expand_count = self.b1 if self.depth == 0 else self.b2
        self.lamb = 0.9
        self.gamma = 0.9
        # self.depth = 0 if parent is None else parent.depth + 1
        self.policy_prob = policy_prob
        self.remote_rm_url = remote_rm_url
        self.action_sequence = action_sequence

    def select(self, cpuct):
        if not self.children:
            return None
        best_value = -float('inf')
        best_node = None
        for action, child in self.children.items():
            # policy_prob = get_policy_prob(child.prompt, child.state, child.model)
            policy_prob = child.policy_prob
            transition_prob = policy_prob / len(action) ** self.lamb
            puct = child.q_values.get(action, 0) + cpuct * transition_prob * np.sqrt(self.visits) / (child.visits + 1)
            if puct > best_value:
                best_value = puct
                best_node = child
            # if not child.terminal and puct > best_value:
            #     best_value = puct
            #     best_node = child
        return best_node
    
    def expand(self):
        current_breadth = self.b1 if self.depth == 0 else self.b2
        current_depth = self.depth
        if len(self.children.items()) >= current_breadth or current_depth >= self.max_depth:
            return
        # n = 1
        # Expand the current node by generating new children
        # prompt = [{"role": "user", "content": f"{self.prompt} {self.state}".strip()}]
        # format_prompt = [
        #     {"role": "user", "content": self.prompt},
        #     {"role": "assistant", "content": self.state}
        # ]
        # prompt = self.tokenizer.apply_chat_template(format_prompt, tokenize=False, add_generation_prompt=False)
        prompt = self.prompt + self.state
        inputs = self.tokenizer(prompt, return_tensors='pt')
        inputs = {k: v.to(torch.cuda.current_device()) for k, v in inputs.items()}

        # responses = self.model.generate(**inputs, deterministic=False)
        inputs_length = inputs["input_ids"].shape[-1]
        prompts_length = len(self.prompt) + len(self.state)
        self.model.model.reset_action_info()
        outputs = self.model.model.generate(
            **inputs,
            use_cache=True,
            max_new_tokens=additional_length,
            do_sample=False,
            top_p=1.0,
            early_stopping=True,
            num_beams=1,
            temperature=0.1,
            repetition_penalty=1.0,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        action_idx, action_prob = self.model.model.get_action_info(prob=True)
        action_idx = action_idx.reshape(-1).cpu().numpy().tolist()
        policy_prob = torch.exp(torch.log(action_prob).sum()).cpu().item()  # bs == 1
        self.model.model.reset_action_info()
        response = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        outputs = outputs[..., inputs_length:]
        action = response[0][prompts_length:]
        
        terminal = is_terminal(outputs[0].cpu().numpy().tolist(), self.tokenizer.eos_token_id) or self.depth >= self.max_depth

        # print("\n\n\n ----maybe new node-----", self.depth, action, terminal)
        
        if action not in self.children:
            print("expanding new node", self.depth, len(self.children.items()), terminal)
            new_node = Node(prompt=self.prompt, state=f"{self.state} {action}", action=action, golden_answer=self.golden_answer, terminal=terminal, model=self.model, tokenizer=self.tokenizer, parent=self, b1=self.b1, b2=self.b2, policy_prob=policy_prob, remote_rm_url=self.remote_rm_url, action_sequence=self.action_sequence+action_idx)
            # print("\n\n\n ----new node-----", self.depth, action, terminal)
            
            self.children[action] = new_node
            new_node.rollout()
            new_node.backup(new_node.gamma)
            return
        else:
            pass
            # print("\n\n\n ----old node-----", self.depth, action, terminal)
    
    def rollout(self):
        self.R = 0.
        # Evaluate the generated result
        # question = self.prompt
        # analysis = self.state
        # golden_answer = self.golden_answer
        # format_prompt = [
        #     {"role": "user", "content": question.strip()},
        #     {"role": "assistant", "content": analysis.strip()}
        # ]
        # prompt = self.tokenizer.apply_chat_template(format_prompt, tokenize=False, add_generation_prompt=False)
        prompt = self.prompt + self.state
        inputs = self.tokenizer(prompt, return_tensors='pt')
        inputs = {k: v.to(torch.cuda.current_device()) for k, v in inputs.items()}

        # responses = self.model.generate(**inputs, deterministic=False)
        self.model.model.reset_action_info()
        outputs = self.model.model.generate(
            **inputs,
            use_cache=True,
            max_new_tokens=max_length,
            do_sample=False,
            top_p=1.0,
            early_stopping=True,
            num_beams=1,
            temperature=0.1,
            repetition_penalty=1.0,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        
        self.model.model.reset_action_info()
        response = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        prompts_length = len(self.prompt) + len(self.state)
        results = response[0][prompts_length:]
        queries = [{"user": self.prompt, "assistant": self.state + results}]

        reward = remote_rm_fn(self.remote_rm_url, queries=queries).to(device=outputs.device)

        ### TODO get remote reward
        reward = reward.reshape(-1).item()

        ### TODO extract answer
        answer = None
        self.correct = 0.
        # check_answer(answer, golden_answer)

        ### TODO get env reward
        env_reward = self.correct

        self.R = reward + env_reward
        return self.R

    def backup(self, gamma):
        self.visits += 1
        # true_count, false_count = 0, 0
        if self.children:
            nume, deno = 0, 0
            for action, child in self.children.items():
                reward = child.R - self.R
                self.q_values[action] = reward + gamma * child.value
                nume += self.q_values[action] * child.visits
                deno += child.visits
                # true_count += child.correct == 1
                # false_count += child.correct == -1
            if nume and deno:
                self.value = nume / deno
        else:
            self.value = self.R

        # if self.children:
        #     if true_count > false_count:
        #         self.correct = 1
        #     elif true_count < false_count:
        #         self.correct = -1
        #     else:
        #         self.correct = 0
        self.correct = 0.

        if self.parent is not None:
            self.parent.backup(gamma)

    def get_response(self):
        items = {}
        items["prompt"] = self.prompt
        items["golden_answer"] = self.golden_answer
        items["response"] = None
        items["maybe_response"] = None
        items["best_value"] = -1000000
        items["maybe_best_value"] = -1000000
        items["action_idx"] = []
        items["maybe_action_idx"] = []
        best_actions = {}
        queue = deque()
        queue.append(self)
        count = 0
        while queue:
            count += 1
            node = queue.popleft()
            if node.children:
                for action, child in node.children.items():
                    if node.q_values.get(action, items["maybe_best_value"]) > items["maybe_best_value"]:
                        items["maybe_best_value"] = node.q_values.get(action, 0)
                        items["maybe_response"] = child.state.strip()
                        items["maybe_action_idx"] = child.action_sequence
                    if child.terminal and node.q_values.get(action, items["best_value"]) > items["best_value"]:
                        items["best_value"] = node.q_values.get(action, 0)
                        items["response"] = child.state.strip()
                        items["action_idx"] = child.action_sequence
                    queue.append(child)
        print("node has steps", count)
        return items

    

    # def get_response(self):
    #     items = {}
    #     items["prompt"] = self.prompt
    #     items["golden_answer"] = self.golden_answer
    #     items["nodes"] = []
    #     best_actions = {}
    #     queue = deque()
    #     queue.append(self)
    #     count = 0
    #     while queue:
    #         count += 1
    #         node = queue.popleft()
    #         cur_dir = {}
    #         cur_dir["value"] = node.value
    #         cur_dir["depth"] = node.depth
    #         cur_dir["terminal"] = node.terminal
    #         cur_dir["correct"] = node.correct
    #         cur_dir["q_values"] = node.q_values
    #         cur_dir["state"] = node.state.strip()
    #         cur_dir["visits"] = node.visits
    #         cur_dir["R"] = node.R
    #         items["nodes"].append(cur_dir)
    #         if node.children:
    #             for action, child in node.children.items():
    #                 if best_actions.get(node.depth) is None or best_actions.get(node.depth)["q_value"] < node.q_values.get(action, 0):
    #                     best_actions[node.depth] = {"state": child.state.strip(), "q_value": node.q_values.get(action, 0)}
    #                 queue.append(child)
    #     items["best_actions"] = best_actions
    #     print("node has steps", count)
    #     return items