source /workspace/chengxing/anaconda3/etc/profile.d/conda.sh
conda activate irl

cd /workspace/chengxing/intentionrl/openrlhf
export PYTHONPATH=$PYTHONPATH:/workspace/chengxing/intentionrl/openrlhf

# data_type="NuminaMath-CoT"
# data_path="/workspace/chengxing/datasets/AI-MO/${data_type}"
data_type="openai-prm800k-solutions-only"
data_path="/workspace/chengxing/datasets/Birchlabs/${data_type}"
sft_path="./checkpoint/NuminaMath-CoT/intention-3.1-8b-v1-sft-dt"
save_path="./checkpoint/${data_type}/intention-3.1-8b-v1-grpo"


deepspeed --include localhost:0,1,2,3,4,5,6,7 --module openrlhf.cli.train_grpo_intention_math \
  --pretrain $sft_path \
  --remote_rm_url "http://172.16.65.222:8000/get_reward" \
  --save_path $save_path \
  --save_steps -1 \
  --logging_steps 1 \
  --eval_steps -1 \
  --micro_train_batch_size 1 \
  --train_batch_size 256 \
  --micro_rollout_batch_size 1 \
  --rollout_batch_size 64 \
  --group_size 4 \
  --max_epochs 1 \
  --num_episodes 1 \
  --prompt_max_len 1024 \
  --generate_max_len 2048 \
  --zero_stage 2 \
  --bf16 \
  --actor_learning_rate 5e-7 \
  --critic_learning_rate 9e-6 \
  --init_kl_coef 0.001 \
  --prompt_data $data_path \
  --input_key instruction \
  --max_samples 8000 \
  --input_template "User: {}\nAssistant: " \
  --normalize_reward \
  --adam_offload \
  --flash_attn \
  --gradient_checkpointing