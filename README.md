Data: 目前使用的是tiger数据集进行训练

Model Structure:
模型结构
模型存放在new_models文件夹中，里面有两个文件，proj.py 和 model.py。前者是用于连接xlmr与llama的转换层，后者是是存放整个模型的地方。其中，xlmr和llama是直接读取已有的数据进行生成，proj的Transformer部分是通过自定义的xlmr config进行生成，Linear Layer部分是自己生成。目前暂时删除了proj中的Transformer部分，进保留Linear Layer部分。

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

在new_models/model.py中，第89行与107行加入了with torch.no_grad().直接运行上面的训练语句，即可进行训练，然后就会出现之前的错误结果。

<img width="868" alt="image" src="https://github.com/CC-Vltava/enlarge_llama/assets/84649088/17406952-2356-4aaf-9782-a8a37e02ba12">

<img width="1273" alt="image" src="https://github.com/CC-Vltava/enlarge_llama/assets/84649088/203dd09f-55a7-43e5-8b8b-339864243773">
