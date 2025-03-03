source /workspace/chengxing/anaconda3/etc/profile.d/conda.sh
conda activate irl

cd /workspace/chengxing/intentionrl/openrlhf
export PYTHONPATH=$PYTHONPATH:/workspace/chengxing/intentionrl/openrlhf


# ["academy", "business", "entertainment", "literature"]
domain="literature"
save_path="./checkpoint/dsp-${domain}/intention3.1-8b-v1-sft-stage1"
data_path="/workspace/chengxing/datasets/dsp/${domain}"

deepspeed --include localhost:4,5,6,7 --module openrlhf.cli.train_sft_intention \
   --max_len 512 \
   --dataset $data_path \
   --input_key prompt \
   --output_key response \
   --input_template '' \
   --train_batch_size 256 \
   --micro_train_batch_size 4 \
   --max_samples 500000 \
   --pretrain /workspace/chengxing/models/intention/intention_model_v1 \
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
   --train_model
