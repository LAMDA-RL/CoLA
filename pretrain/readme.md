# Data download

We recommand download the datasets via hfd tools:

```
./hfd.sh dataname --dataset --tool aria2c -x 16 --local-dir datapath --local-dir-use-symlinks False --hf_token yourtoken --hf_username yourname
```

# Model download and process

For Llama-3.1 8B

```
python litgpt/scripts/download.py --repo_id meta-llama/Meta-Llama-3.1-8B --model_name meta-llama/Meta-Llama-3.1-8B
```

# Data process

For the four dataset (Slimpajama, Starcoder, Proofpile, Wudao), process by

```
python litgpt/data/prepare_proofpile.py --input_dir yourdatapath --tokenizer_path yourtokenizerpath
python litgpt/data/prepare_slimpajama.py --input_dir yourdatapath --tokenizer_path yourtokenizerpath
python litgpt/data/prepare_starcoder.py --input_dir yourdatapath --tokenizer_path yourtokenizerpath
python litgpt/data/prepare_wudao.py --input_dir yourdatapath --tokenizer_path yourtokenizerpath
```

# Training

For two stage training on multi-nodes:

```
export PYTHONPATH=$PYTHONPATH:your main path
# stage 1
/usr/local/bin/fabric run \
--node-rank=$your node index \
--main-port=$your main port \
--main-address=$your main address \
--accelerator=cuda \
--devices=8 \
--num-nodes=4 \
litgpt/pretrain_intention_multi.py --num_nodes 4

# stage 2
/usr/local/bin/fabric run \
--node-rank=$your node index \
--main-port=$your main port \
--main-address=$your main address \
--accelerator=cuda \
--devices=8 \
--num-nodes=4 \
litgpt/pretrain_policy_multi.py --num_nodes 4
```

For two stage training on one-node:

```
export PYTHONPATH=$PYTHONPATH:your main path
# stage 1
python ./litgpt/pretrain_intention.py

# stage 2
python ./litgpt/pretrain_policy.py
```

# Converting lit checkpoint to huggingface style

After training on large scale datasets, converting the ckpt to huggingface model

```
python litgpt/scripts/convert_lit_checkpoint.py --checkpoint_dir yourckptpath --output_dir youroutputpath
```
