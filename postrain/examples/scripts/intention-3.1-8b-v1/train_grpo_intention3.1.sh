source /workspace/chengxing/anaconda3/etc/profile.d/conda.sh
conda activate irl

cd /workspace/chengxing/intentionrl/openrlhf
export PYTHONPATH=$PYTHONPATH:/workspace/chengxing/intentionrl/openrlhf

# ["academy", "business", "entertainment", "literature"]
domain="business"
sft_path="./checkpoint/dsp-${domain}/intention3.1-8b-v1-sft-stage2"
rm_path="./checkpoint/dsp-${domain}/llama3.1-8b-rm"
save_path="./checkpoint/dsp-${domain}/intention3.1-8b-v1-grpo-lr1e5"
data_path="/workspace/chengxing/datasets/dsp/${domain}"

deepspeed --include localhost:0,1,2,3,4,5,6,7 --module openrlhf.cli.train_grpo_intention \
  --pretrain $sft_path \
  --reward_pretrain $rm_path \
  --save_path $save_path \
  --save_steps -1 \
  --logging_steps 1 \
  --eval_steps -1 \
  --micro_train_batch_size 2 \
  --train_batch_size 256 \
  --micro_rollout_batch_size 16 \
  --rollout_batch_size 1024 \
  --group_size 16 \
  --max_epochs 1 \
  --prompt_max_len 256 \
  --generate_max_len 256 \
  --zero_stage 2 \
  --bf16 \
  --actor_learning_rate 1e-5 \
  --critic_learning_rate 9e-6 \
  --init_kl_coef 0.01 \
  --prompt_data $data_path \
  --input_key prompt \
  --max_samples 100000 \
  --input_template '' \
  --normalize_reward \
  --adam_offload \
  --flash_attn \
  --gradient_checkpointing
