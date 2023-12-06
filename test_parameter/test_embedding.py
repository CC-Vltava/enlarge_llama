
from transformers import AutoTokenizer, LlamaForCausalLM, LlamaTokenizer, AutoModel
import torch
from accelerate import Accelerator
import os
import torch.nn as nn


# llama path: /home/cchuan/Project/MiniGPT-4/weight/vicuna/
# /data1/cchuan/data/weight/tiny_llama

PATH_TO_CONVERTED_MODEL="/data1/cchuan/data/weight/tiny_llama"

PATH_TO_CONVERTED_TOKENIZER='/data1/cchuan/data/weight/tiny_llama'


tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

prompt = "I like cats"

num = tokenizer.add_special_tokens({
    "bos_token": "<s>",
    "eos_token": "</s>",
    "unk_token": "<unk>",
    "pad_token": "<pad>",
})

assert num <= 1, 'error'
tokenizer.padding_side='left'
inputs1 = tokenizer(
    prompt + tokenizer.eos_token,
    return_tensors='pt',
    max_length=30, 
    truncation=True,
    padding='max_length',
)

inputs2 = tokenizer(
    prompt + tokenizer.eos_token,
    return_tensors='pt',
    max_length=30, 
    truncation=True,
    padding='longest',
)

embedding_layer = nn.Embedding(num_embeddings=32001, embedding_dim=256, padding_idx=None)

output1 = embedding_layer(inputs1.input_ids)
output2 = embedding_layer(inputs2.input_ids)

p = 1