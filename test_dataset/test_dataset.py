import argparse
import logging
import math
import os
import random
import datasets
from datasets import Dataset
import torch
from functools import partial
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from model.model import GPT
import json
from datasets import load_dataset
from langdetect import detect

import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaTokenizer,
    LlamaTokenizerFast,
    SchedulerType,
    DataCollatorForSeq2Seq,
    get_scheduler,
    GPTNeoXTokenizerFast,
    GPT2Tokenizer,
    OPTForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training


def encode_with_prompt_completion_format(example, encode_tokenizer, decode_tokenizer, max_seq_length):
    '''
    Here we assume each example has 'prompt' and 'completion' fields.
    We concatenate prompt and completion and tokenize them together because otherwise prompt will be padded/trancated 
    and it doesn't make sense to follow directly with the completion.
    '''
    # if prompt doesn't end with space and completion doesn't start with space, add space
    # print('here!!!-------------****************')
    # print(example)

    input_ids1 = encode_tokenizer(
        example['prompt'], 
        return_tensors='pt', 
        max_length=max_seq_length, 
        truncation=True,
        padding='longest'
    )['input_ids']

    input_ids2 = decode_tokenizer(
        example['completion'] + decode_tokenizer.eos_token, 
        return_tensors='pt', 
        max_length=max_seq_length, 
        truncation=True,
        padding='longest'
    )['input_ids']

    labels = decode_tokenizer(
        example['completion'] + decode_tokenizer.eos_token, 
        return_tensors='pt', 
        max_length=max_seq_length, 
        truncation=True,
        padding='longest'
    )['input_ids']
    
    pad_length = input_ids1.shape[1]

    pad_labels = torch.ones((labels.shape[0], pad_length), dtype=labels.dtype) * -100

    labels = torch.cat([pad_labels, labels], dim=1)

    attention_mask = torch.ones_like(input_ids1)

    assert labels.shape[1] == input_ids1.shape[1] + input_ids2.shape[1], 'error!'

    def is_english(sentence):
        try:
            return detect(sentence) == 'en'
        except:
            # 如果无法确定语言，当作非英文处理
            return False
    
    return {
        'input_ids': input_ids1.flatten(),
        'labels': labels.flatten(),
        'attention_mask': attention_mask.flatten(),
        'sec_input_ids': input_ids2.flatten(),
        'language': is_english(example['prompt']) and is_english(example['completion']),
        'input': example['prompt'],
        'output': example['completion']
    }



def read_data():
    data_list = []

    data_path = '/data1/cchuan/data/mllm/clean_data1.json'

    with open(data_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            data_list.append(data)
    
    print('raw data size')
    print(len(data_list[0]['train']))

    formatted_data = {
        "train": [{"prompt": item["input"], "completion": item["output"]} for item in data_list[0]['train'][: 30]],
        "test": [{"prompt": item["input"], "completion": item["output"]} for item in data_list[0]['train'][: 30]],
    }

    save_data_path = '/data1/cchuan/output_file.json'

    with open(save_data_path, 'w') as json_file:
        json.dump(formatted_data, json_file, indent=4)
    
    print('finish reading')

    return load_dataset('json', data_files=save_data_path, field='train')


def collate_fn(batch):
    # batch是一个包含单个样本的列表，每个样本是一个字典{'input': input_sequence, 'labels': target_sequence}

    # 提取输入和标签
    inputs = [item['input_ids'] for item in batch]
    labels = [item['labels'] for item in batch]
    atts = [item['attention_mask'] for item in batch]
    input2 = [item['sec_input_ids'] for item in batch]
    # ori_input = [item['input'] for item in batch]
    # ori_output = [item['output'] for item in batch]
    # return batch
    return {
        'input_ids': torch.stack(inputs), 
        'labels': torch.stack(labels), 
        'attention_mask': torch.stack(atts),
        'sec_input_ids':  torch.stack(input2)
    }



encode_tokenizer = AutoTokenizer.from_pretrained('/data1/cchuan/data/weight/xlmr/')
decode_tokenizer = AutoTokenizer.from_pretrained('/data1/cchuan/data/weight/tiny_llama/')

decode_tokenizer.add_special_tokens({
    "bos_token": "<s>",
    "eos_token": "</s>",
    "unk_token": "<unk>",
    "pad_token": "<pad>",
})
raw_datasets = read_data()
encode_function = partial(
    encode_with_prompt_completion_format,
    encode_tokenizer=encode_tokenizer,
    decode_tokenizer=decode_tokenizer,
    max_seq_length=256,
)


# print('The size of raw data')
# print(len(raw_datasets['train']))
lm_datasets = raw_datasets.map(
    encode_function,
    batched=False,
    num_proc=16,
    remove_columns=[name for name in raw_datasets["train"].column_names if name not in ["input_ids", "labels", "attention_mask"]],
    desc="Tokenizing and reformatting instruction data",
)
lm_datasets.set_format(type="pt")
lm_datasets = lm_datasets.filter(lambda example: (example['language']))
lm_datasets = lm_datasets.filter(lambda example: (example['labels'] != -100).any())
print('The size of clean data')
print(len(lm_datasets['train']))

train_dataset = lm_datasets['train']

train_dataloader = DataLoader(
    train_dataset, 
    shuffle=True, 
    # collate_fn=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding="longest"),
    collate_fn=collate_fn,
    batch_size=1
)

# for batch in train_dataloader:
#     # 在这里对每个批次的数据进行操作
#     # batch 是一个包含样本的列表或元组，具体取决于数据集和collate_fn的设置
#     # 例如，如果 batch_size=1，那么 batch[0] 就是当前样本
#     current_sample = batch[0]
#     input_ids = current_sample['input_ids']
#     labels = current_sample['labels']
#     input = current_sample['input']
#     output = current_sample['output']
#     true_input_ids = encode_tokenizer(
#         input, 
#         return_tensors='pt', 
#         max_length=256, 
#         truncation=True,
#         padding='longest'
#     )['input_ids']
#     true_output_ids = decode_tokenizer(
#         output + decode_tokenizer.eos_token, 
#         return_tensors='pt', 
#         max_length=256, 
#         truncation=True,
#         padding='longest'
#     )['input_ids']
#     # print(input_ids.sum() - true_input_ids.sum() - true_output_ids.sum())
    
#     # 在这里执行您的操作...


model = GPT()


for batch in train_dataloader:
    model(**batch)