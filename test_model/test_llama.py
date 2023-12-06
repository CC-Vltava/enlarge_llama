from test_model_llama.model import GPT
import torch
from transformers import AutoTokenizer, LlamaForCausalLM, LlamaTokenizer, AutoModel
import torch
from accelerate import Accelerator
import os
import torch.nn as nn


PATH_TO_CONVERTED_MODEL="/data1/cchuan/data/weight/tiny_llama"
PATH_TO_CONVERTED_TOKENIZER='/data1/cchuan/data/weight/xlmr'


encode_tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)
decode_tokenizer =  AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_MODEL)
xlmr_model = AutoModel.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)
llama_model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_MODEL)
# xlmr1 = AutoModel.from_pretrained(PATH_TO_CONVERTED_MODEL)

model = GPT()

prompt = "I like cats, but I do not like dogs!"

num = decode_tokenizer.add_special_tokens({
    "bos_token": "<s>",
    "eos_token": "</s>",
    "unk_token": "<unk>",
    "pad_token": "<pad>",
})

model.resize_token_embeddings(len(decode_tokenizer))
llama_model.resize_token_embeddings(len(decode_tokenizer))

inputs = encode_tokenizer(
    prompt,
    return_tensors='pt',
    max_length=30, 
    truncation=True,
    padding='max_length',
)
sec_input_ids = decode_tokenizer(
    prompt,
    return_tensors='pt',
    max_length=30, 
    truncation=True,
    padding='max_length',
)


output1 = model(**inputs, sec_input_ids=sec_input_ids.input_ids)


llama_input_ids = model.proj(xlmr_model(**inputs).last_hidden_state)
print(sec_input_ids.input_ids)
sec_input_embed = llama_model.model.embed_tokens(sec_input_ids.input_ids)\
    .to(llama_input_ids.dtype)
llama_input_embed = torch.cat([llama_input_ids, sec_input_embed], dim=1)
shape = llama_input_embed.shape
attention_mask = torch.zeros([shape[0], shape[1]])
attention_mask[:, :llama_input_ids.shape[1]] = 1
print('output2')
print('3')
print(llama_input_ids)
print('4')
print(sec_input_embed)
output2 = llama_model(inputs_embeds=llama_input_embed, attention_mask=attention_mask)




p = 1