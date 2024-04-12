export CUDA_VISIBLE_DEVICES=0

MODEL_SIZE=7B
NUM_GPUS=1
BATCH_SIZE_PER_GPU=1
TOTAL_BATCH_SIZE=256
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --deepspeed_config_file ds_configs/stage2.conf \
    open_instruct/finetune.py \
    --model_name_or_path ../hf_llama_models/${MODEL_SIZE} \
    --use_flash_attn \
    --tokenizer_name ../hf_llama_models/${MODEL_SIZE} \
    --use_slow_tokenizer \
    --train_file /data1/cchuan/tulu/test_tulu.json \
    --max_seq_length 512 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 1e-4 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0.01 \
    --num_train_epochs 8 \
    --output_dir /data1/cchuan/new_weight/2batch_tiger/test_GPT_${MODEL_SIZE}/ \
    --with_tracking \
    --report_to tensorboard \
    --logging_steps 1
