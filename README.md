# Controlling large langauge model with Latent Action (Llama-3.1-CoLA)

This repository provides the pre-trained inverse dynamics models, world models and BC policy model of CoLA, based on Llama-3.1-8B-base:
* **Language World Model**: This model has an action space of 64 actions and generates language based on these actions.
* **Inverse Dynamics Model**: This model derives actions from input sentences, essentially "understanding" the language.
* **Policy Model**: This model is training to output actions that simulate the language data used in training.

The Llama-3.1-10B-CoLA model is the model of paper "Controlling Large Language Model with Latent Action", which follows the similar learning process as the BWArea model in the paper "BWArea Model: Learning World Model, Inverse Dynamics, and Policy for Controllable Language Generation." (https://huggingface.co/jiacx0229/intention_pretrained_2.7B_30G). However, it is based on a pre-trained autoregressive model to provide pre-trained embeddings. In this version, we use the embeddings from the final block of Llama-3.1-8B as the input embedding for each module in CoLA. Compared to the BWArea model with 2.7B parameters, the CoLA model has a larger parameter size of 11B, which includes 8B for Llama-3.1-8B, 1B for the inverse dynamics model and world model, and 2B for the policy model. Additionally, the CoLA model has been pre-trained on a larger dataset with nearly 1.1T tokens, encompassing general corpora, math, and code. The corpus includes both English and Chinese.

## Dataset

The dataset used in this project is sourced from the following repositories:
1. Slimpajama (https://huggingface.co/datasets/cerebras/SlimPajama-627B)
2. Starcoder (https://huggingface.co/datasets/bigcode/starcoderdata)
3. Math-Proof-Pile-2 (https://huggingface.co/datasets/EleutherAI/proof-pile-2)
4. Wudao (https://huggingface.co/datasets/p208p2002/wudao)

The dataset totally contains 1.1T tokens. For inverse dynamics model and language world model, they are trained on the whole dataset and for the policy model, the released version is trained on 640G tokens of dataset.


## Latent Action Control

The CoLA model is designed to extract language actions from a pre-trained model. The extracted latent actions represent distinct semantic meanings of language, providing a higher-level generation compared to the token space, where the combination of tokens may not necessarily form a valid sentence. We hypothesize that generating sentences using latent action tokens can result in greater diversity and higher quality compared to token-space sampling. 

To evaluate this, we employ the **BGE-M3 model** (https://huggingface.co/BAAI/bge-m3), which outputs sentence embeddings to compute semantic similarity between different sentences. Semantic similarity serves as a metric to assess quality diversity. When the semantic similarity between generated sentences is higher, it indicates lower quality diversity. While random token sampling can produce very different sequences, this does not necessarily result in lower semantic similarity because it often fails to generate valid sentences.

To compute quality diversity, we follow these steps:
1. Use **N sequences** from the validation dataset as prefixes.
2. Generate **M sentences** using different methods for each prefix.
3. Each set of **M sentences** forms a group, resulting in **N groups** in total.
4. Compute the semantic similarity within each group using the **BGE-M3 model** in an each-to-each manner.
5. Average these values across all groups to obtain the final quality diversity metric.
6. Take the reciprocal of the semantic similarity values as the quality diversity value.

The results are as follows:

| Method                                   | Quality Diversity (Avg) | Semantic Similarity (Avg) |
|------------------------------------------|-------------------------|---------------------------|
| Latent Action Sampling (Llama-3.1-CoLA)  | 1.92                    | 0.52                      |
| Token Space Sampling (Llama-3.1-base)    | 1.54                    | 0.65                      |
| Random Token Sampling                    | 1.30                    | 0.77                      |

where Latent Action Sampling is using our world model and random action tokens to generate sentence, Token Sapce Sampling is using Lllama-3.1-8B-base logits to generate and Random Token Sampling is random token sequence. This show that the latent action control can derive a higher quality diversity (QD).

We also show that as the training tokens growing, CoLA's world model can generate sentences with larger QD:

| Method                            | Quality Diversity (Avg) | Semantic Similarity (Avg) |
|-----------------------------------|-------------------------|---------------------------|
| Latent Action Sampling (CoLA-1.1T)| 1.92                    | 0.52                      |
| Latent Action Sampling (CoLA-10G) | 1.67                    | 0.60                      |
| Latent Action Sampling (CoLA-1G)  | 1.45                    | 0.69                      |

where 1.1T, 10G and 1G are the number of training tokens, implies the growing QD with larger number of tokens.

For BC policy model, we evaluate its performance on serveral benchmarks, including MMLU (general ability), MathQA (math ability) and CMMLU (Chinese ability).

| Method                 | MMLU  | MathQA | CMMLU |
|------------------------|-------|--------|-------|
| BC Policy Model (640G) | 65.39 | 37.02  | 52.24 |

## Install

```
pip install -r requirements.txt
```

## Usage

### Loading Model and Tokenizer

```python
import torch
from transformers import AutoTokenizer
from intention import IntentionForCausalLM
# load model and tokenizer
model_path = "./models/intention/intention_model_idmonly"   # your path
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
model = IntentionForCausalLM.from_pretrained(model_path, torch_dtype="auto")   
```

### Use Language World Model for Language Generation

```python
# The language world model take actions as input and generate the next token.
# In this example, you can try different actions and see how the language world model generates
examples = "I like eating"
# using world model under selected action
action_idx = 3 # choose your action between 0 to 63
inputs = tokenizer(prompt, return_tensors='pt')
inputs = {k: v for k, v in inputs.items()}
logits = model.forward_world_model(
    **inputs,
    action_idx=action_idx,
)
next_token = logits[:, -1].argmax(dim=1, keepdim=True)
```

### Use Policy Model to Select Actions

```python
# The policy model was pretrained according to the training data.
# This example shows the actions of the pre-trained policy
model.set_action_sampling(greedy=False, tau=2.0)  # greedy=True for deterministic action, tau for temperature of action 
prompt = "I like eating"
inputs = tokenizer(prompt, return_tensors='pt')
inputs = {k: v for k, v in inputs.items()}  # intention model
action_logits = model(**inputs, act_mode=True)
action_index = action_logits[, -1].argmax(dim=1, keepdim=True) 
```

### Sentence Generation

```python
# This example uses the CoLA as a common LLM for language generation.
model.set_action_sampling(greedy=False, tau=2.0)  # greedy=True for deterministic action, tau for temperature of action 
prompt = "I like eating"
inputs = tokenizer(prompt, return_tensors='pt')
inputs = {k: v for k, v in inputs.items()}  # intention model

with torch.no_grad():
    model.reset_action_info()
    outputs = model.generate(
        **inputs,
        use_cache=True,
        max_new_tokens=8,
        do_sample=False,
        top_p=1.0,
        early_stopping=True,
        num_beams=1,
        temperature=0.1,
        repetition_penalty=1.0,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    action_idx, _ = model.get_action_info(prob=True)
    action_idx = action_idx.reshape(-1).numpy().tolist()  # get action sequence during generation
    model.reset_action_info()
    results = tokenizer.batch_decode(outputs, skip_special_tokens=True)

```

## Acknowledgements

- (https://github.com/Lightning-AI/litgpt/)
- (https://github.com/OpenRLHF/OpenRLHF)


