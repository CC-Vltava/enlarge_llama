
from transformers import AutoTokenizer, LlamaForCausalLM, LlamaTokenizer, AutoModel
import torch
from accelerate import Accelerator
import os


# llama path: /home/cchuan/Project/MiniGPT-4/weight/vicuna/
# /data1/cchuan/data/weight/tiny_llama

PATH_TO_CONVERTED_MODEL="/data1/cchuan/data/weight/tiny_llama"
PATH_TO_CONVERTED_TOKENIZER='/data1/cchuan/data/weight/tiny_llama'

model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_MODEL)
tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

prompt = "Some information about Northwestern University"

num = tokenizer.add_special_tokens({
    "bos_token": "<s>",
    "eos_token": "</s>",
    "unk_token": "<unk>",
    "pad_token": "<pad>",
})

assert num <= 1, 'error'

embedding_size = model.get_input_embeddings().weight.shape[0]
if len(tokenizer) > embedding_size:
    model.resize_token_embeddings(len(tokenizer))

print('finish load model')


# 特殊符号之间添加空格
inputs = tokenizer(
    prompt + tokenizer.eos_token,
    return_tensors='pt',
    max_length=30, 
    truncation=True,
    padding='max_length',
)


# 进行embedding
inputs_embeds = model.model.embed_tokens(inputs.input_ids)
print(inputs_embeds.shape)


# Generate
print('max + att')
generate_ids = model.generate(**inputs, max_length=60)
output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(output)

inputs = tokenizer(
    prompt + tokenizer.eos_token,
    return_tensors='pt',
    max_length=30, 
    truncation=True,
    padding='longest'
)

print('lognest + att')
generate_ids = model.generate(**inputs, max_length=60)
output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(output)



print('lognest')
generate_ids = model.generate(input_ids=inputs.input_ids, max_length=60)
output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(output)


print('不同长度对于输出影响')
print('30')
inputs = tokenizer(
    prompt + tokenizer.eos_token,
    return_tensors='pt',
    max_length=30, 
    truncation=True,
    padding='max_length'
)
generate_ids = model.generate(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, max_length=120)
output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(output)
print('80')
inputs = tokenizer(
    prompt + tokenizer.eos_token,
    return_tensors='pt',
    max_length=80, 
    truncation=True,
    padding='max_length'
)
generate_ids = model.generate(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, max_length=120)
output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(output)
