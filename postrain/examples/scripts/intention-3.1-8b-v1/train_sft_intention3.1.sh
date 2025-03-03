source /workspace/chengxing/anaconda3/etc/profile.d/conda.sh
conda activate irl

cd /workspace/chengxing/intentionrl/openrlhf
export PYTHONPATH=$PYTHONPATH:/workspace/chengxing/intentionrl/openrlhf

data_type="NuminaMath-CoT"
data_path="/workspace/chengxing/datasets/AI-MO/${data_type}"
model_path="/workspace/chengxing/models/intention/intention_model_v1_new"
save_path="./checkpoint/${data_type}/intention-3.1-8b-v1-sft-dt"

deepspeed --include localhost:0,1,2,3,4,5,6,7 --module openrlhf.cli.train_sft \
   --max_len 2048 \
   --dataset $data_path \
   --input_key problem \
   --output_key solution \
   --input_template "User: {}\nAssistant: " \
   --train_batch_size 256 \
   --micro_train_batch_size 4 \
   --max_samples 1000000 \
   --pretrain $model_path \
   --save_path $save_path \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --zero_stage 2 \
   --max_epochs 1 \
   --bf16 \
   --flash_attn \
   --learning_rate 5e-6 \
   --gradient_checkpointing \
   --intention_model