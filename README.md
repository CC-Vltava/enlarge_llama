### Enlarge LLaMA Model

Data:
数据存储在'/data1/cchuan/data/mllm/clean_data1.json'中，已经整理

Dataset:
在./open_instruct/finetune.py中，写有调用对应Dataset的函数read_data()
因为这个数据集较小而且处于测试阶段，所以现在只写了batch_size=1的情况，就无需额外加入padding等操作
然后在后续进一步进行处理之后转为Dataloader

Model Structure:
模型结构
模型存放在new_models文件夹中，里面有两个文件，proj.py 和 model.py。前者是用于连接xlmr与llama的转换层，后者是是存放整个模型的地方。其中，xlmr和llama是直接读取已有的数据进行生成，proj的Transformer部分是通过自定义的xlmr config进行生成，Linear Layer部分是自己生成。

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


