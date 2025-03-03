source /workspace/chengxing/anaconda3/etc/profile.d/conda.sh
conda activate irl

cd /workspace/chengxing/intentionrl/openrlhf
export PYTHONPATH=$PYTHONPATH:/workspace/chengxing/intentionrl/openrlhf

deepspeed --module openrlhf.cli.train_ppo_intention \
  --pretrain ./checkpoint/full-hh-rlhf/intention3.1-8b-v1-sft-stage2 \
  --critic_pretrain ./checkpoint/full-hh-rlhf/llama3.1-8b-sft \
  --reward_pretrain ./checkpoint/full-hh-rlhf/llama3.1-8b-rm \
  --save_path ./checkpoint/full-hh-rlhf/intention3.1-8b-v1-ppo \
  --save_steps -1 \
  --logging_steps 1 \
  --eval_steps -1 \
  --micro_train_batch_size 1 \
  --train_batch_size 128 \
  --micro_rollout_batch_size 8 \
  --rollout_batch_size 256 \
  --max_epochs 1 \
  --prompt_max_len 256 \
  --generate_max_len 256 \
  --zero_stage 2 \
  --bf16 \
  --actor_learning_rate 5e-6 \
  --critic_learning_rate 9e-6 \
  --init_kl_coef 0.01 \
  --prompt_data /workspace/chengxing/datasets/Dahoas/full-hh-rlhf \
  --input_key prompt \
  --max_samples 20000 \
  --input_template '' \
  --normalize_reward \
  --adam_offload \
  --flash_attn \
  --gradient_checkpointing

# --apply_chat_template \