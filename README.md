运行./scripts/finetune_with_accelerate.sh > output_test.log 2>&1进行训练

Train:
训练使用的是./open_instruct/finetune.py文件 然后这个是使用./scripts/finetune_with_accelerate.sh进行训练 

如果需要调节训练参数，可以在./scripts/finetune_with_accelerate.sh进行调整，其中这几个参数因为在原有模型中写死了，所以调节并没有用
```
    --model_name_or_path ../hf_llama_models/${MODEL_SIZE} \
    --use_flash_attn \
    --tokenizer_name ../hf_llama_models/${MODEL_SIZE} \
    --use_slow_tokenizer \
    --train_file data/processed/tulu_v1/tulu_v1_data.jsonl \
```
