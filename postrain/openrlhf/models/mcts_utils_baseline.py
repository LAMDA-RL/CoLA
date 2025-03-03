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
    def __init__(self, model, qmodel, tokenizer, cpuct=0.7, gamma=0.9, b1=5, b2=4, T=32, D=64, remote_rm_url=None):
        self.model = model
        self.qmodel = qmodel
        self.tokenizer = tokenizer
        self.cpuct = cpuct
        self.gamma = gamma
        self.b1 = b1
        self.b2 = b2
        self.T = T
        self.D = D
        self.leaves = []
        self.remote_rm_url = remote_rm_url
        
    def search(self, prompt, golden_answer, is_print=False):
        root = Node(prompt=prompt, state=prompt, action=[-1], content=prompt, golden_answer=golden_answer, terminal=False, sentence_terminal=False, model=self.model, qmodel=self.qmodel, tokenizer=self.tokenizer, parent=None, b1=self.b1, b2=self.b2, policy_prob=0.5, remote_rm_url=self.remote_rm_url, action_sequence=[], max_depth=self.D)
        
        for round in range(self.T):
            # print("round", round)
            node = root
            if is_print:
                print("select")
            selected_node = node.select(self.cpuct, is_print=is_print)
            if selected_node is None:
                break
            
            max_tries = 10
            tries = 0
            while (not selected_node.terminal) and (len(selected_node.children.items()) < selected_node.expand_count):
                if is_print:
                    print(round, "expand")
                terminal = selected_node.expand(is_print=is_print)
                tries += 1
                if tries >= max_tries:
                    break
                if tries > selected_node.expand_count:
                    print("warning!!! try {}".format(tries))
            if terminal:
                print("term")
                break
        return root.get_response(is_print=is_print)
    

class Node:
    def __init__(self, prompt, state, action, content, golden_answer, terminal, sentence_terminal, model, qmodel, tokenizer, parent=None, b1=5, b2=4, policy_prob=0.5, remote_rm_url=None, action_sequence=[], max_depth=64, new_depth=1):
        # self.id = time.time()
        self.prompt = prompt
        self.state = state
        self.action = action
        self.content = content
        self.max_depth = max_depth
        self.terminal = terminal
        self.sentence_terminal = sentence_terminal
        self.golden_answer = golden_answer
        self.model = model
        self.qmodel = qmodel
        self.tokenizer = tokenizer
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.value = 0
        self.q_values = {}
        self.depth = parent.depth + new_depth if parent else 0
        self.R = 0
        self.b1 = b1
        self.correct = 0
        self.b2 = b2
        self.expand_count = self.b1 if self.depth == 0 else self.b2
        self.lamb = 0.9
        self.gamma = 0.99
        # self.depth = 0 if parent is None else parent.depth + 1
        self.policy_prob = policy_prob
        self.remote_rm_url = remote_rm_url
        self.action_sequence = action_sequence

        self.action_logits = None
        self.policy_embeds = None
        self.action_values = None
        self.qmodel_kv_cache = None
        self.wmodel_kv_cache = None

    def select(self, cpuct, is_print=False):
        # if not self.children:
        #     return None
        
        all_leaf_nodes = []
        queue = deque()
        queue.append([self, self.visits])
        count = 0
        max_depth = 0
        # print(len(queue), "???")
        while queue:
            if is_print:
                print("queue")
            count += 1
            node, visit = queue.popleft()
            max_depth = node.depth if node.depth > max_depth else max_depth
            # print((len(node.children.items()) < node.expand_count))
            # print(not node.terminal)
            # print("???????????????")
            if (not node.terminal) and (len(node.children.items()) < node.expand_count):
                all_leaf_nodes.append([node, visit])
            if node.children:
                for action, child in node.children.items():
                    queue.append([child, node.visits])
                    
        depth_thres = 100
        best_value = -float('inf')
        best_node = None
        for obj in all_leaf_nodes:
            node, visits = obj[0], obj[1]
            policy_prob = node.policy_prob
            transition_prob = policy_prob  #  / len(action) ** self.lamb
            puct = node.value + cpuct * transition_prob * np.sqrt(visits) / (node.visits + 1)
            # if puct > best_value:
            #     best_value = puct
            #     best_node = child
            if is_print:
                print("node depth {}, max depth {}, terminal {}".format(node.depth, max_depth, node.terminal))
            if not node.terminal and puct > best_value and node.depth >= max_depth - depth_thres:
                best_value = puct
                best_node = node
        return best_node
    
    def get_dynaq(self, inputs):
        # self.model.model.reset_action_info()
        # compute current Qsa
        if self.action_logits is None:
            outputs = self.model.model(
                **inputs,
                # ea_mode=True,
            )
            qoutputs = self.qmodel(
                **inputs
            )
            logits = outputs.logits[:, -1, :]
            self.action_logits = logits.cpu()
            # self.policy_embeds = policy_embeds.cpu()
            self.action_values = qoutputs.cpu()
        else:
            qoutputs = self.action_values.to(torch.cuda.current_device())
            logits = self.action_logits.to(torch.cuda.current_device())
            # print(logits)
        action_prob = F.softmax(logits, dim=-1)
        action_prob, action_idx = action_prob.max(dim=-1)
        action_prob = action_prob.item()
        
        qvalue = qoutputs[:, -1, action_idx].item()
        self.action_logits[:, action_idx] = -99999
        next_token = action_idx.reshape(-1, 1)  # next_logits.argmax(dim=-1)
        
        inputs["input_ids"] = torch.cat([inputs["input_ids"], next_token], dim=-1)
        inputs["attention_mask"] = torch.cat([inputs["attention_mask"], torch.ones_like(next_token).type(inputs["attention_mask"].dtype)], dim=-1)
        
        action_idx = action_idx.item()
        

        next_qvalue = self.qmodel(**inputs)[:, -1].max().item()

        return {
            "qvalue": qvalue,
            "action_idx": action_idx,
            "action_prob": action_prob,
            "next_token": next_token,
            "next_qvalue": next_qvalue,
        }, inputs
    
    # def get_dynaq(self, inputs):
    #     self.model.model.reset_action_info()
    #     # compute current Qsa
    #     outputs = self.model.model(
    #         **inputs,
    #         ea_mode=True,
    #     )
    #     qoutputs = self.qmodel(
    #         **inputs
    #     )
    #     logits, policy_embeds = outputs[0][:, -1, :], outputs[1]
    #     action_prob = F.gumbel_softmax(logits, dim=-1)
    #     action_prob, action_idx = action_prob.max(dim=-1)
    #     action_prob = action_prob.item()
    #     action_idx = action_idx.item()
    #     qvalue = qoutputs[:, -1, action_idx].item()
    #     # action_logits[:, action_idx] = -99999
    #     next_logits = self.model.model.forward_world_model(
    #         **inputs,
    #         policy_embeds=policy_embeds,
    #         action_idx=action_idx,
    #     )
    #     next_token = next_logits.argmax(dim=-1)

    #     inputs["input_ids"] = torch.cat([inputs["input_ids"], next_token], dim=-1)
    #     inputs["attention_mask"] = torch.cat([inputs["attention_mask"], torch.ones_like(next_token).type(inputs["attention_mask"].dtype)], dim=-1)

    #     next_qvalue = self.qmodel(**inputs)[:, -1].max().item()

    #     return {
    #         "qvalue": qvalue,
    #         "action_idx": action_idx,
    #         "action_prob": action_prob,
    #         "next_token": next_token,
    #         "next_qvalue": next_qvalue,
    #     }, inputs
    
    def expand(self, is_print=False):
        current_breadth = self.b1 if self.depth == 0 else self.b2
        current_depth = self.depth
        if len(self.children.items()) >= current_breadth or current_depth >= self.max_depth:
            return
        prompt = self.content
        # print("prompt\n\n", prompt, "\n\nprompt\n\n")
        inputs = self.tokenizer(prompt, return_tensors='pt')
        inputs = {k: v.to(torch.cuda.current_device()) for k, v in inputs.items()}
        
        unc_terminal = True
        terminal = False
        action_idx = []
        action_prob = 0.0
        step_count = 0
        # while unc_terminal and not terminal:
        # get dynamics and qvalue
        outputs, inputs = self.get_dynaq(inputs)
        # print(inputs["input_ids"].shape)
        action_idx += [outputs["action_idx"]]
        # print("expand", self.depth, outputs["action_idx"])
        reward = 0.
        response = self.tokenizer.batch_decode(inputs["input_ids"], skip_special_tokens=True)  
        
        prompts_length = len(self.prompt)
        results = response[0][prompts_length:] 
        uncertainty = (outputs["qvalue"] - (reward + self.gamma * outputs["next_qvalue"])) ** 2
        # uncertainty = 0.0
        # print(uncertainty, outputs["qvalue"])
        unc_terminal = uncertainty < 0.01
        action_prob += outputs["action_prob"]
        step_count += 1
        terminal = (outputs["next_token"].reshape(-1).item() == self.tokenizer.eos_token_id)
        action_label = outputs["action_idx"]
        # forward if terminal
        if unc_terminal and not terminal:
            # responses = self.model.generate(**inputs, deterministic=False)
            # self.model.model.reset_action_info()
            # self.model.model.model.deterministic = False
            # print(inputs["input_ids"].shape)
            outputs_unc = self.model.model.generate(
                **inputs,
                use_cache=True,
                max_new_tokens=64,
                do_sample=True,
                top_p=1.0,
                early_stopping=True,
                num_beams=1,
                temperature=0.5,
                repetition_penalty=1.0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            # print(outputs_unc.shape)
            # idx, prob = self.model.model.get_action_info(prob=True)
            idx = outputs_unc[:, inputs["input_ids"].shape[1]:].reshape(-1).cpu().numpy().tolist()
            # prob = prob.reshape(-1).sum().item()
            action_idx += idx
            step_count += len(idx)
            outputs["next_token"] = outputs_unc[:, -1:]  
            # self.model.model.reset_action_info()
            
            outs = outputs_unc.reshape(-1).tolist()
            response = self.tokenizer.batch_decode(outputs_unc, skip_special_tokens=True)
            # print(outs, "outs")
            for token in outs:
                if token == self.tokenizer.eos_token_id:
                    terminal = True
                    break
        
        action_prob = action_prob / step_count 
        # print("node action length : {}".format(step_count))
        terminal = (outputs["next_token"].reshape(-1).item() == self.tokenizer.eos_token_id)
        
        true_terminal = terminal or self.depth >= self.max_depth
        # print("tt", true_terminal)
        # print("dt", self.depth >= self.max_depth)
        # if len(action_idx) > 1 or str(action_idx) not in [key for key in self.children.keys()]:
            # print("\n\n\nnew node", response[0], "\n\n\nnew node \n\n\n")
        new_node = Node(prompt=self.prompt, state=response[0], action=action_idx, content=response[0], golden_answer=self.golden_answer, terminal=true_terminal, sentence_terminal=terminal, model=self.model, qmodel=self.qmodel, tokenizer=self.tokenizer, parent=self, b1=self.b1, b2=self.b2, policy_prob=action_prob, remote_rm_url=self.remote_rm_url, action_sequence=self.action_sequence+action_idx, max_depth=self.max_depth, new_depth=step_count)

        # self.q_values[outputs["action_idx"]] = outputs["qvalue"] #- uncertainty
        self.children[action_label] = new_node
        # simulation value
        # new_node.R = outputs["next_qvalue"] if not terminal else reward
        new_node.rollout()
        new_node.backup(new_node.gamma)
        return terminal
        # if is_print:
        #     print("fail expanding", self.depth)

    def rollout(self):
        self.R = 0.
        
        prompt = self.content  # self.prompt + self.state
        inputs = self.tokenizer(prompt, return_tensors='pt')
        inputs = {k: v.to(torch.cuda.current_device()) for k, v in inputs.items()}
        input_length = inputs["input_ids"].shape[-1]

        # responses = self.model.generate(**inputs, deterministic=False)
        # self.model.model.reset_action_info()
        # self.model.model.model.deterministic = False
        outputs = self.model.model.generate(
            **inputs,
            use_cache=True,
            max_new_tokens=max_length,
            do_sample=True,
            top_p=1.0,
            early_stopping=True,
            num_beams=1,
            temperature=0.5,
            repetition_penalty=1.0,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        
        # self.model.model.reset_action_info()
        response = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        self.state = response[0]
        prompts_length = len(self.prompt)   # + len(self.state)
        results = response[0][prompts_length:]
        # results = ""
        queries = [{"user": self.prompt, "assistant": results}]
        
        # max_content_length = 256
        # content_outputs = outputs[..., input_length:input_length+max_content_length]
        # content_results = self.tokenizer.batch_decode(content_outputs, skip_special_tokens=True)
        
        # self.sentence_terminal = input_length + max_content_length == outputs.shape[-1]
        # self.terminal = True if self.sentence_terminal else self.terminal
        
        # self.content = self.state + content_results[0]
        # print(self.content, "content")
        # print(queries)

        reward = remote_rm_fn(self.remote_rm_url, queries=queries)

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
                # reward = gamma * child.R - self.R
                # reward = self.R - gamma * child.R
                reward = child.R - self.R
                # print(reward, self.depth, child.terminal)
                # self.q_values[action] = (reward + gamma * child.value) * 0.2 + self.q_values[action] * 0.8
                self.q_values[action] = (reward + gamma * child.value)
                nume += self.q_values[action] * child.visits
                deno += child.visits
                # true_count += child.correct == 1
                # false_count += child.correct == -1
            if nume and deno:
                self.value = nume / deno
        else:
            self.value = self.R

        self.correct = 0.

        if self.parent is not None:
            self.parent.backup(gamma)

    def get_response(self, is_print=False):
        items = {}
        items["prompt"] = self.prompt
        items["golden_answer"] = self.golden_answer
        items["response"] = None
        items["maybe_response"] = None
        items["best_value"] = -1000000
        items["maybe_best_value"] = -1000000
        items["action_idx"] = []
        items["maybe_action_idx"] = []
        # items["true_term"] = False
        best_actions = {}
        queue = deque()
        queue.append(self)
        count = 0
        depth = 0
        maybe_depth = 0
        true_terminal = False

        node_list = []
        first_flag = True
        while queue:
            count += 1
            node = queue.popleft()
            if not first_flag:
                node_list.append(node)
            else:
                first_flag = False
            if node.children:
                for action, child in node.children.items():
                    child.value = node.q_values.get(action, child.value)
                    queue.append(child)
        if is_print:
            print("len node", len(node_list))
        for node in reversed(node_list):
            if node.R > items["maybe_best_value"]:
                items["maybe_best_value"] = node.R
                items["maybe_response"] = node.state
                items["maybe_action_idx"] = node.action_sequence
                # items["true_term"] = False
        threshold = 0.0
        # for node in reversed(node_list):
        #     if node.terminal and node.value > items["best_value"] and node.value > items["maybe_best_value"] - threshold:
        #         items["best_value"] = node.value
        #         items["response"] = node.content
        #         items["action_idx"] = node.action_sequence
        #         depth = node.depth
        #         true_terminal = node.sentence_terminal
                # items["true_term"] = True if true_terminal else False
        if items["response"] is None:
            items["best_value"] = items["maybe_best_value"]
            items["response"] = items["maybe_response"]
            items["action_idx"] = items["maybe_action_idx"]

        # while queue:
        #     count += 1
        #     node = queue.popleft()
        #     if node.children:
        #         for action, child in node.children.items():
                    # if is_print:
                    #     print(child.terminal)
                    # print(node.q_values.get(action, items["maybe_best_value"]) > items["maybe_best_value"], node.depth, node.q_values.get(action, items["maybe_best_value"]), items["maybe_best_value"], node.terminal, node.sentence_terminal)
                    # if node.q_values.get(action, items["maybe_best_value"]) > items["maybe_best_value"] and child.depth >= maybe_depth:
                    #     items["maybe_best_value"] = node.q_values.get(action, 0)
                    #     # items["maybe_response"] = child.state.strip() #+ " " + child.content.strip()
                    #     items["maybe_response"] = child.content
                    #     items["maybe_action_idx"] = child.action_sequence
                    #     maybe_depth = child.depth
                    # if child.terminal and node.q_values.get(action, items["best_value"]) > items["best_value"] and child.depth >= depth:
                    #     items["best_value"] = node.q_values.get(action, 0)
                    #     # items["response"] = child.state.strip() #+ " " + child.content.strip()
                    #     items["response"] = child.content
                    #     items["action_idx"] = child.action_sequence
                    #     depth = child.depth
                    #     true_terminal = child.sentence_terminal
                    # queue.append(child)
                    # # print(child.value, node.q_values.get(action, items["maybe_best_value"]), child.depth, child.terminal, child.sentence_terminal)
        if is_print:
            print("node has steps", count, depth, maybe_depth)
            # assert 0
        items["true_terminal"] = true_terminal
        return items 





# class MCTS:
#     def __init__(self, model, qmodel, tokenizer, cpuct=0.7, gamma=0.9, b1=5, b2=4, T=32, remote_rm_url=None):
#         self.model = model
#         self.qmodel = qmodel
#         self.tokenizer = tokenizer
#         self.cpuct = cpuct
#         self.gamma = gamma
#         self.b1 = b1
#         self.b2 = b2
#         self.T = T
#         self.leaves = []
#         self.remote_rm_url = remote_rm_url
        
#     def search(self, prompt, golden_answer):
#         root = Node(prompt=prompt, state="", action=-1, content="", golden_answer=golden_answer, terminal=False, model=self.model, qmodel=self.qmodel, tokenizer=self.tokenizer, parent=None, b1=self.b1, b2=self.b2, policy_prob=0.5, remote_rm_url=self.remote_rm_url, action_sequence=[])
#         for round in range(self.T):
#             # print("round", round)
#             node = root
#             while not node.terminal:
#                 if (not node.terminal) and (len(node.children.items()) < node.expand_count):
#                     node.expand()
#                     break
#                 selected_node = node.select(self.cpuct)
#                 if selected_node is None:
#                     break
#                 node = selected_node
#         return root.get_response()
    

# class Node:
#     def __init__(self, prompt, state, action, content, golden_answer, terminal, model, qmodel, tokenizer, parent=None, b1=5, b2=4, policy_prob=0.5, remote_rm_url=None, action_sequence=[]):
#         # self.id = time.time()
#         self.prompt = prompt
#         self.state = state
#         self.action = action
#         self.content = content
#         self.max_depth = 64
#         self.terminal = terminal
#         self.golden_answer = golden_answer
#         self.model = model
#         self.qmodel = qmodel
#         self.tokenizer = tokenizer
#         self.parent = parent
#         self.children = {}
#         self.visits = 0
#         self.value = 0
#         self.q_values = {}
#         self.depth = parent.depth + 1 if parent else 0
#         self.R = 0
#         self.b1 = b1
#         self.correct = 0
#         self.b2 = b2
#         self.expand_count = self.b1 if self.depth == 0 else self.b2
#         self.lamb = 0.9
#         self.gamma = 0.99
#         # self.depth = 0 if parent is None else parent.depth + 1
#         self.policy_prob = policy_prob
#         self.remote_rm_url = remote_rm_url
#         self.action_sequence = action_sequence

#         self.action_logits = None
#         self.action_values = None
#         self.qmodel_kv_cache = None
#         self.wmodel_kv_cache = None

#     def select(self, cpuct):
#         if not self.children:
#             return None
#         best_value = -float('inf')
#         best_node = None
#         for action, child in self.children.items():
#             # policy_prob = get_policy_prob(child.prompt, child.state, child.model)
#             policy_prob = child.policy_prob
#             transition_prob = policy_prob  #  / len(action) ** self.lamb
#             puct = child.q_values.get(action, 0) + cpuct * transition_prob * np.sqrt(self.visits) / (child.visits + 1)
#             # if puct > best_value:
#             #     best_value = puct
#             #     best_node = child
#             if not child.terminal and puct > best_value:
#                 best_value = puct
#                 best_node = child
#         return best_node
    
#     def get_dynaq(self, inputs):
#         # self.model.model.reset_action_info()
#         # compute current Qsa
#         if self.action_logits is None:
#             outputs = self.model(
#                 **inputs,
#                 return_output=True
#             )
#             qoutputs = self.qmodel(
#                 **inputs
#             )
#             logits = outputs.logits
#             self.action_logits = logits.cpu()
#             self.action_values = qoutputs.cpu()
#         else:
#             qoutputs = self.action_values.to(torch.cuda.current_device())
#             logits = self.action_logits.to(torch.cuda.current_device())
#             # print(logits)
#         action_prob = F.gumbel_softmax(logits, dim=-1)
#         action_prob, action_idx = action_prob.max(dim=-1)
#         action_prob = action_prob.item()
#         action_idx = action_idx.item()
#         qvalue = qoutputs[:, -1, action_idx].item()
#         self.action_logits[:, action_idx] = -99999

#         next_token = action_idx
#         inputs["input_ids"] = torch.cat([inputs["input_ids"], next_token], dim=-1)
#         inputs["attention_mask"] = torch.cat([inputs["attention_mask"], torch.ones_like(next_token).type(inputs["attention_mask"].dtype)], dim=-1)

#         next_qvalue = self.qmodel(**inputs)[:, -1].max().item()

#         return {
#             "qvalue": qvalue,
#             "action_idx": action_idx,
#             "action_prob": action_prob,
#             "next_token": next_token,
#             "next_qvalue": next_qvalue,
#         }, inputs
    
#     def expand(self):
#         current_breadth = self.b1 if self.depth == 0 else self.b2
#         current_depth = self.depth
#         if len(self.children.items()) >= current_breadth or current_depth >= self.max_depth:
#             return
#         prompt = self.prompt + self.state
#         inputs = self.tokenizer(prompt, return_tensors='pt')
#         inputs = {k: v.to(torch.cuda.current_device()) for k, v in inputs.items()}

#         # get dynamics and qvalue
#         outputs, inputs = self.get_dynaq(inputs)
#         print("expand", self.depth, outputs["action_idx"])
#         terminal = is_terminal(inputs["input_ids"].cpu().numpy().tolist(), self.tokenizer.eos_token_id)

#         response = self.tokenizer.batch_decode(inputs["input_ids"], skip_special_tokens=True)
#         prompts_length = len(self.prompt) + len(self.state)
#         results = response[0][prompts_length:]
#         if terminal:
#             queries = [{"user": self.prompt, "assistant": self.state + results}]
#             reward = remote_rm_fn(self.remote_rm_url, queries=queries).reshape(-1).item()
#         else:
#             reward = 0.

#         uncertainty = (outputs["qvalue"] - (reward + self.gamma * outputs["next_qvalue"])) ** 2
#         # print(uncertainty, outputs["qvalue"])
#         unc_terminal = uncertainty < 0.008
#         true_terminal = terminal or unc_terminal or self.depth >= self.max_depth
#         if outputs["action_idx"] not in self.children:
#             new_node = Node(prompt=self.prompt, state=f"{self.state} {results}", action=outputs["action_idx"], content="", golden_answer=self.golden_answer, terminal=true_terminal, model=self.model, qmodel=self.qmodel, tokenizer=self.tokenizer, parent=self, b1=self.b1, b2=self.b2, policy_prob=outputs["action_prob"], remote_rm_url=self.remote_rm_url, action_sequence=self.action_sequence+[outputs["action_idx"]])

#             self.q_values[outputs["action_idx"]] = outputs["qvalue"]
#             new_node.visits = 1
#             self.children[outputs["action_idx"]] = new_node
#             if terminal:
#                 new_node.R = reward
#             else:
#                 new_node.rollout()
#             new_node.backup(new_node.gamma)
#             return
#         print("fail expanding", self.depth)
    
#     def rollout(self):
#         self.R = 0.
#         # Evaluate the generated result
#         # question = self.prompt
#         # analysis = self.state
#         # golden_answer = self.golden_answer
#         # format_prompt = [
#         #     {"role": "user", "content": question.strip()},
#         #     {"role": "assistant", "content": analysis.strip()}
#         # ]
#         # prompt = self.tokenizer.apply_chat_template(format_prompt, tokenize=False, add_generation_prompt=False)
#         prompt = self.prompt + self.state
#         inputs = self.tokenizer(prompt, return_tensors='pt')
#         inputs = {k: v.to(torch.cuda.current_device()) for k, v in inputs.items()}

#         # responses = self.model.generate(**inputs, deterministic=False)
#         # self.model.model.reset_action_info()
#         outputs = self.model.model.generate(
#             **inputs,
#             use_cache=True,
#             max_new_tokens=max_length,
#             do_sample=True,
#             top_p=1.0,
#             early_stopping=True,
#             num_beams=1,
#             temperature=0.1,
#             repetition_penalty=1.0,
#             pad_token_id=self.tokenizer.pad_token_id,
#             eos_token_id=self.tokenizer.eos_token_id,
#         )
        
#         # self.model.model.reset_action_info()
#         response = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
#         prompts_length = len(self.prompt) + len(self.state)
#         results = response[0][prompts_length:]
#         self.content = results
#         queries = [{"user": self.prompt, "assistant": self.state + results}]
#         # print(queries)

#         reward = remote_rm_fn(self.remote_rm_url, queries=queries).to(device=outputs.device)

#         ### TODO get remote reward
#         reward = reward.reshape(-1).item()

#         ### TODO extract answer
#         answer = None
#         self.correct = 0.
#         # check_answer(answer, golden_answer)

#         ### TODO get env reward
#         env_reward = self.correct

#         self.R = reward + env_reward
#         return self.R

#     def backup(self, gamma):
#         self.visits += 1
#         # true_count, false_count = 0, 0
#         if self.children:
#             nume, deno = 0, 0
#             for action, child in self.children.items():
#                 reward = child.R - self.R
#                 self.q_values[action] = reward + gamma * child.value
#                 nume += self.q_values[action] * child.visits
#                 deno += child.visits
#                 # true_count += child.correct == 1
#                 # false_count += child.correct == -1
#             if nume and deno:
#                 self.value = nume / deno
#         else:
#             self.value = self.R

#         # if self.children:
#         #     if true_count > false_count:
#         #         self.correct = 1
#         #     elif true_count < false_count:
#         #         self.correct = -1
#         #     else:
#         #         self.correct = 0
#         self.correct = 0.

#         if self.parent is not None:
#             self.parent.backup(gamma)

#     def get_response(self):
#         items = {}
#         items["prompt"] = self.prompt
#         items["golden_answer"] = self.golden_answer
#         items["response"] = None
#         items["maybe_response"] = None
#         items["best_value"] = -1000000
#         items["maybe_best_value"] = -1000000
#         items["action_idx"] = []
#         items["maybe_action_idx"] = []
#         best_actions = {}
#         queue = deque()
#         queue.append(self)
#         count = 0
#         while queue:
#             count += 1
#             node = queue.popleft()
#             if node.children:
#                 for action, child in node.children.items():
#                     if node.q_values.get(action, items["maybe_best_value"]) > items["maybe_best_value"]:
#                         items["maybe_best_value"] = node.q_values.get(action, 0)
#                         items["maybe_response"] = child.state.strip() + " " + child.content.strip()
#                         items["maybe_action_idx"] = child.action_sequence
#                     if child.terminal and node.q_values.get(action, items["best_value"]) > items["best_value"]:
#                         items["best_value"] = node.q_values.get(action, 0)
#                         items["response"] = child.state.strip() + " " + child.content.strip()
#                         items["action_idx"] = child.action_sequence
#                     queue.append(child)
#         print("node has steps", count)
#         return items


# import numpy as np
# import torch
# import torch.nn.functional as F
# from collections import deque
# from openrlhf.utils.remote_rm_utils import remote_rm_fn, remote_rm_fn_ray


# additional_length = 128
# max_length = 1024
# input_template = "User: {}\nAssistant: " 
# def is_terminal(action, eos_id):
#     return eos_id in action

# class MCTS:
#     def __init__(self, model, qmodel, tokenizer, cpuct=0.7, gamma=0.9, b1=5, b2=4, T=32, remote_rm_url=None):
#         self.model = model
#         self.qmodel = qmodel
#         self.tokenizer = tokenizer
#         self.cpuct = cpuct
#         self.gamma = gamma
#         self.b1 = b1
#         self.b2 = b2
#         self.T = T
#         self.leaves = []
#         self.remote_rm_url = remote_rm_url
        
#     def search(self, prompt, golden_answer):
#         root = Node(prompt=prompt, state="", action=-1, content="", golden_answer=golden_answer, terminal=False, model=self.model, qmodel=self.qmodel, tokenizer=self.tokenizer, parent=None, b1=self.b1, b2=self.b2, policy_prob=0.5, remote_rm_url=self.remote_rm_url, action_sequence=[])
#         for round in range(self.T):
#             # print("round", round)
#             node = root
#             while not node.terminal:
#                 if (not node.terminal) and (len(node.children.items()) < node.expand_count):
#                     node.expand()
#                     break
#                 selected_node = node.select(self.cpuct)
#                 if selected_node is None:
#                     break
#                 node = selected_node
#         return root.get_response()
    

# class Node:
#     def __init__(self, prompt, state, action, content, golden_answer, terminal, model, qmodel, tokenizer, parent=None, b1=5, b2=4, policy_prob=0.5, remote_rm_url=None, action_sequence=[]):
#         # self.id = time.time()
#         self.prompt = prompt
#         self.state = state
#         self.action = action
#         self.content = content
#         self.max_depth = 64
#         self.terminal = terminal
#         self.golden_answer = golden_answer
#         self.model = model
#         self.qmodel = qmodel
#         self.tokenizer = tokenizer
#         self.parent = parent
#         self.children = {}
#         self.visits = 0
#         self.value = 0
#         self.q_values = {}
#         self.depth = parent.depth + 1 if parent else 0
#         self.R = 0
#         self.b1 = b1
#         self.correct = 0
#         self.b2 = b2
#         self.expand_count = self.b1 if self.depth == 0 else self.b2
#         self.lamb = 0.9
#         self.gamma = 0.99
#         # self.depth = 0 if parent is None else parent.depth + 1
#         self.policy_prob = policy_prob
#         self.remote_rm_url = remote_rm_url
#         self.action_sequence = action_sequence

#         self.action_logits = None
#         self.policy_embeds = None
#         self.action_values = None
#         self.qmodel_kv_cache = None
#         self.wmodel_kv_cache = None

#     def select(self, cpuct):
#         if not self.children:
#             return None
#         best_value = -float('inf')
#         best_node = None
#         for action, child in self.children.items():
#             # policy_prob = get_policy_prob(child.prompt, child.state, child.model)
#             policy_prob = child.policy_prob
#             transition_prob = policy_prob  #  / len(action) ** self.lamb
#             puct = child.q_values.get(action, 0) + cpuct * transition_prob * np.sqrt(self.visits) / (child.visits + 1)
#             # if puct > best_value:
#             #     best_value = puct
#             #     best_node = child
#             if not child.terminal and puct > best_value:
#                 best_value = puct
#                 best_node = child
#         return best_node
    
#     def get_dynaq(self, inputs):
#         self.model.model.reset_action_info()
#         # compute current Qsa
#         if self.action_logits is None:
#             outputs = self.model.model(
#                 **inputs,
#                 ea_mode=True,
#             )
#             qoutputs = self.qmodel(
#                 **inputs
#             )
#             logits, policy_embeds = outputs[0][:, -1, :], outputs[1]
#             self.action_logits = logits.cpu()
#             self.policy_embeds = policy_embeds.cpu()
#             self.action_values = qoutputs.cpu()
#         else:
#             qoutputs = self.action_values.to(torch.cuda.current_device())
#             logits, policy_embeds = self.action_logits.to(torch.cuda.current_device()), self.policy_embeds.to(torch.cuda.current_device())
#             # print(logits)
#         action_prob = F.gumbel_softmax(logits, dim=-1)
#         action_prob, action_idx = action_prob.max(dim=-1)
#         action_prob = action_prob.item()
#         action_idx = action_idx.item()
#         qvalue = qoutputs[:, -1, action_idx].item()
#         self.action_logits[:, action_idx] = -99999

#         next_logits = self.model.model.forward_world_model(
#             **inputs,
#             policy_embeds=policy_embeds,
#             action_idx=action_idx,
#         )
#         next_token = next_logits.argmax(dim=-1)

#         inputs["input_ids"] = torch.cat([inputs["input_ids"], next_token], dim=-1)
#         inputs["attention_mask"] = torch.cat([inputs["attention_mask"], torch.ones_like(next_token).type(inputs["attention_mask"].dtype)], dim=-1)

#         next_qvalue = self.qmodel(**inputs)[:, -1].max().item()

#         return {
#             "qvalue": qvalue,
#             "action_idx": action_idx,
#             "action_prob": action_prob,
#             "next_token": next_token,
#             "next_qvalue": next_qvalue,
#         }, inputs
    
#     def expand(self):
#         current_breadth = self.b1 if self.depth == 0 else self.b2
#         current_depth = self.depth
#         if len(self.children.items()) >= current_breadth or current_depth >= self.max_depth:
#             return
#         prompt = self.prompt + self.state
#         inputs = self.tokenizer(prompt, return_tensors='pt')
#         inputs = {k: v.to(torch.cuda.current_device()) for k, v in inputs.items()}

#         # get dynamics and qvalue
#         outputs, inputs = self.get_dynaq(inputs)
#         print("expand", self.depth, outputs["action_idx"])
#         terminal = is_terminal(inputs["input_ids"].cpu().numpy().tolist(), self.tokenizer.eos_token_id)

#         response = self.tokenizer.batch_decode(inputs["input_ids"], skip_special_tokens=True)
#         prompts_length = len(self.prompt) + len(self.state)
#         results = response[0][prompts_length:]
#         if terminal:
#             queries = [{"user": self.prompt, "assistant": self.state + results}]
#             reward = remote_rm_fn(self.remote_rm_url, queries=queries).reshape(-1).item()
#         else:
#             reward = 0.

#         uncertainty = (outputs["qvalue"] - (reward + self.gamma * outputs["next_qvalue"])) ** 2
#         # print(uncertainty, outputs["qvalue"])
#         unc_terminal = uncertainty < 0.008
#         true_terminal = terminal or unc_terminal or self.depth >= self.max_depth
#         if outputs["action_idx"] not in self.children:
#             new_node = Node(prompt=self.prompt, state=f"{self.state} {results}", action=outputs["action_idx"], content="", golden_answer=self.golden_answer, terminal=true_terminal, model=self.model, qmodel=self.qmodel, tokenizer=self.tokenizer, parent=self, b1=self.b1, b2=self.b2, policy_prob=outputs["action_prob"], remote_rm_url=self.remote_rm_url, action_sequence=self.action_sequence+[outputs["action_idx"]])

#             self.q_values[outputs["action_idx"]] = outputs["qvalue"]
#             new_node.visits = 1
#             self.children[outputs["action_idx"]] = new_node
#             if terminal:
#                 new_node.R = reward
#             else:
#                 new_node.rollout()
#             new_node.backup(new_node.gamma)
#             return
#         print("fail expanding", self.depth)
    
#     def rollout(self):
#         self.R = 0.
#         # Evaluate the generated result
#         # question = self.prompt
#         # analysis = self.state
#         # golden_answer = self.golden_answer
#         # format_prompt = [
#         #     {"role": "user", "content": question.strip()},
#         #     {"role": "assistant", "content": analysis.strip()}
#         # ]
#         # prompt = self.tokenizer.apply_chat_template(format_prompt, tokenize=False, add_generation_prompt=False)
#         prompt = self.prompt + self.state
#         inputs = self.tokenizer(prompt, return_tensors='pt')
#         inputs = {k: v.to(torch.cuda.current_device()) for k, v in inputs.items()}

#         # responses = self.model.generate(**inputs, deterministic=False)
#         self.model.model.reset_action_info()
#         outputs = self.model.model.generate(
#             **inputs,
#             use_cache=True,
#             max_new_tokens=max_length,
#             do_sample=False,
#             top_p=1.0,
#             early_stopping=True,
#             num_beams=1,
#             temperature=0.1,
#             repetition_penalty=1.0,
#             pad_token_id=self.tokenizer.pad_token_id,
#             eos_token_id=self.tokenizer.eos_token_id,
#         )
        
#         self.model.model.reset_action_info()
#         response = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
#         prompts_length = len(self.prompt) + len(self.state)
#         results = response[0][prompts_length:]
#         self.content = results
#         queries = [{"user": self.prompt, "assistant": self.state + results}]
#         # print(queries)

#         reward = remote_rm_fn(self.remote_rm_url, queries=queries).to(device=outputs.device)

#         ### TODO get remote reward
#         reward = reward.reshape(-1).item()

#         ### TODO extract answer
#         answer = None
#         self.correct = 0.
#         # check_answer(answer, golden_answer)

#         ### TODO get env reward
#         env_reward = self.correct

#         self.R = reward + env_reward
#         return self.R

#     def backup(self, gamma):
#         self.visits += 1
#         # true_count, false_count = 0, 0
#         if self.children:
#             nume, deno = 0, 0
#             for action, child in self.children.items():
#                 reward = child.R - self.R
#                 self.q_values[action] = reward + gamma * child.value
#                 nume += self.q_values[action] * child.visits
#                 deno += child.visits
#                 # true_count += child.correct == 1
#                 # false_count += child.correct == -1
#             if nume and deno:
#                 self.value = nume / deno
#         else:
#             self.value = self.R

#         # if self.children:
#         #     if true_count > false_count:
#         #         self.correct = 1
#         #     elif true_count < false_count:
#         #         self.correct = -1
#         #     else:
#         #         self.correct = 0
#         self.correct = 0.

#         if self.parent is not None:
#             self.parent.backup(gamma)

#     def get_response(self):
#         items = {}
#         items["prompt"] = self.prompt
#         items["golden_answer"] = self.golden_answer
#         items["response"] = None
#         items["maybe_response"] = None
#         items["best_value"] = -1000000
#         items["maybe_best_value"] = -1000000
#         items["action_idx"] = []
#         items["maybe_action_idx"] = []
#         best_actions = {}
#         queue = deque()
#         queue.append(self)
#         count = 0
#         while queue:
#             count += 1
#             node = queue.popleft()
#             if node.children:
#                 for action, child in node.children.items():
#                     if node.q_values.get(action, items["maybe_best_value"]) > items["maybe_best_value"]:
#                         items["maybe_best_value"] = node.q_values.get(action, 0)
#                         items["maybe_response"] = child.state.strip() + " " + child.content.strip()
#                         items["maybe_action_idx"] = child.action_sequence
#                     if child.terminal and node.q_values.get(action, items["best_value"]) > items["best_value"]:
#                         items["best_value"] = node.q_values.get(action, 0)
#                         items["response"] = child.state.strip() + " " + child.content.strip()
#                         items["action_idx"] = child.action_sequence
#                     queue.append(child)
#         print("node has steps", count)
#         return items

    

#     # def get_response(self):
#     #     items = {}
#     #     items["prompt"] = self.prompt
#     #     items["golden_answer"] = self.golden_answer
#     #     items["nodes"] = []
#     #     best_actions = {}
#     #     queue = deque()
#     #     queue.append(self)
#     #     count = 0
#     #     while queue:
#     #         count += 1
#     #         node = queue.popleft()
#     #         cur_dir = {}
#     #         cur_dir["value"] = node.value
#     #         cur_dir["depth"] = node.depth
#     #         cur_dir["terminal"] = node.terminal
#     #         cur_dir["correct"] = node.correct
#     #         cur_dir["q_values"] = node.q_values
#     #         cur_dir["state"] = node.state.strip()
#     #         cur_dir["visits"] = node.visits
#     #         cur_dir["R"] = node.R
#     #         items["nodes"].append(cur_dir)
#     #         if node.children:
#     #             for action, child in node.children.items():
#     #                 if best_actions.get(node.depth) is None or best_actions.get(node.depth)["q_value"] < node.q_values.get(action, 0):
#     #                     best_actions[node.depth] = {"state": child.state.strip(), "q_value": node.q_values.get(action, 0)}
#     #                 queue.append(child)
#     #     items["best_actions"] = best_actions
#     #     print("node has steps", count)
#     #     return items