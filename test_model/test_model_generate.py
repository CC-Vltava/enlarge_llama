from test_model_generator.model import GPT
import torch
from transformers import AutoTokenizer, LlamaForCausalLM, LlamaTokenizer, AutoModel
import torch
from accelerate import Accelerator
import os
import torch.nn as nn


PATH_TO_CONVERTED_MODEL="/data1/cchuan/data/weight/tiny_llama"
PATH_TO_CONVERTED_TOKENIZER='/data1/cchuan/data/weight/tiny_llama'


tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)
llama_model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_MODEL)
# xlmr1 = AutoModel.from_pretrained(PATH_TO_CONVERTED_MODEL)

model = GPT()

prompt = "I like cats, but I do not like dogs!"

num = tokenizer.add_special_tokens({
    "bos_token": "<s>",
    "eos_token": "</s>",
    "unk_token": "<unk>",
    "pad_token": "<pad>",
})

model.resize_token_embeddings(len(tokenizer))
llama_model.resize_token_embeddings(len(tokenizer))

inputs = tokenizer(
    prompt,
    return_tensors='pt',
    max_length=30, 
    truncation=True,
    padding='longest',
)

# embedding = model.llama_model.model.embed_tokens(inputs.input_ids)

print('output1')
output1 = model.generate(inputs.input_ids)
output = tokenizer.batch_decode(output1, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(output)

print('output2')
output2 = llama_model.generate(**inputs, max_new_tokens=30)
output = tokenizer.batch_decode(output2, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(output)

p=1
