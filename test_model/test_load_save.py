
from transformers import AutoTokenizer, LlamaForCausalLM, LlamaTokenizer, AutoModel
import torch
from accelerate import Accelerator
import os
import torch.nn as nn
from test_model.model import GPT



# llama path: /home/cchuan/Project/MiniGPT-4/weight/vicuna/
# /data1/cchuan/data/weight/tiny_llama

PATH_TO_CONVERTED_TOKENIZER='/data1/cchuan/data/weight/tiny_llama'

tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

model1 = GPT()
model2 = GPT()


prompt = "I like cats"

num = tokenizer.add_special_tokens({
    "bos_token": "<s>",
    "eos_token": "</s>",
    "unk_token": "<unk>",
    "pad_token": "<pad>",
})

assert num <= 1, 'error'
model1.llama_model.resize_token_embeddings(len(tokenizer))
model2.llama_model.resize_token_embeddings(len(tokenizer))

inputs = tokenizer(
    prompt + tokenizer.eos_token,
    return_tensors='pt',
    max_length=30, 
    truncation=True,
    padding='max_length',
)

output1 = model1(**inputs)
output2 = model2(**inputs)


torch.save(model1.proj.state_dict(), './model1_weights.pth')
model2.proj.load_state_dict(torch.load('./model1_weights.pth'))
model2.llama_model.resize_token_embeddings(len(tokenizer))
output4 = model2(**inputs)

p = 1
