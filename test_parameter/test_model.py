
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
model = AutoModel.from_pretrained(PATH_TO_CONVERTED_MODEL)

prompt = "I like cats"

num = tokenizer.add_special_tokens({
    "bos_token": "<s>",
    "eos_token": "</s>",
    "unk_token": "<unk>",
    "pad_token": "<pad>",
})

assert num <= 1, 'error'


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
embedding_size = model.get_input_embeddings().weight.shape[0]
if len(tokenizer) > embedding_size:
    model.resize_token_embeddings(len(tokenizer))



output1 = model(**inputs1)
output2 = model(**inputs2)

embedding_input = model.embed_tokens(inputs1.input_ids)
output3 = model(inputs_embeds=embedding_input, attention_mask=inputs1.attention_mask)

p = 1
