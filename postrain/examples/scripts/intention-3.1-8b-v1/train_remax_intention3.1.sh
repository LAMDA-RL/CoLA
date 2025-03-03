source /workspace/chengxing/anaconda3/etc/profile.d/conda.sh
conda activate irl

cd /workspace/chengxing/intentionrl/openrlhf/
export PYTHONPATH=$PYTHONPATH:/workspace/chengxing/intentionrl/openrlhf/

data_type="openai-prm800k-solutions-only"
data_path="/workspace/chengxing/datasets/Birchlabs/${data_type}"
sft_path="./checkpoint/NuminaMath-CoT/intention-3.1-8b-v1-sft-dt"
save_path="./checkpoint/${data_type}/intention-3.1-8b-v1-remax-t10-kl00001"


deepspeed --include localhost:0,2,3,4 --module openrlhf.cli.train_remax_intention \
  --pretrain $sft_path \
  --remote_rm_url "http://172.16.64.22:8000/get_reward" \
  --save_path $save_path \
  --save_steps -1 \
  --logging_steps 1 \
  --eval_steps -1 \
  --micro_train_batch_size 2 \
  --train_batch_size 256 \
  --micro_rollout_batch_size 8 \
  --rollout_batch_size 256 \
  --max_epochs 1 \
  --prompt_max_len 1024 \
  --generate_max_len 1024 \
  --zero_stage 2 \
  --bf16 \
  --actor_learning_rate 1e-5 \
  --critic_learning_rate 9e-6 \
  --init_kl_coef 0.0001 \
  --prompt_data $data_path \
  --input_key instruction \
  --max_samples 100000 \
  --input_template "User: {}\nAssistant: " \
  --normalize_reward \
  --adam_offload \
  --flash_attn \
  --gradient_checkpointing