source /workspace/chengxing/anaconda3/etc/profile.d/conda.sh
conda activate irl

cd /workspace/chengxing/intentionrl/openrlhf
export PYTHONPATH=$PYTHONPATH:/workspace/chengxing/intentionrl/openrlhf

domain="literature"
save_path="./checkpoint/dsp-${domain}/llama3.1-8b-rm"
data_path="/workspace/chengxing/datasets/dsp/${domain}"

deepspeed --include localhost:4,5,6,7 --module openrlhf.cli.train_rm \
   --save_path $save_path \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --train_batch_size 256 \
   --micro_train_batch_size 4 \
   --pretrain /workspace/chengxing/models/meta-llama/Meta-Llama-3.1-8B \
   --bf16 \
   --max_epochs 1 \
   --max_len 512 \
   --zero_stage 3 \
   --learning_rate 9e-6 \
   --dataset $data_path \
   --input_template '' \
   --prompt_key prompt \
   --chosen_key chosen \
   --rejected_key rejected \
   --flash_attn \
   --gradient_checkpointing