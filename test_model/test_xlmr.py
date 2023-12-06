from test_model_xlmr.model import GPT
import torch
from transformers import AutoTokenizer, LlamaForCausalLM, LlamaTokenizer, AutoModel
import torch
from accelerate import Accelerator
import os
import torch.nn as nn


PATH_TO_CONVERTED_MODEL="/data1/cchuan/data/weight/xlmr"
PATH_TO_CONVERTED_TOKENIZER='/data1/cchuan/data/weight/xlmr'


tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)
xlmr = AutoModel.from_pretrained(PATH_TO_CONVERTED_MODEL)
xlmr1 = AutoModel.from_pretrained(PATH_TO_CONVERTED_MODEL)
model = GPT()

prompt = "I like cats, but I do not like dogs!"

num = tokenizer.add_special_tokens({
    "bos_token": "<s>",
    "eos_token": "</s>",
    "unk_token": "<unk>",
    "pad_token": "<pad>",
})

inputs = tokenizer(
    prompt,
    return_tensors='pt',
    max_length=30, 
    truncation=True,
    padding='max_length',
)

output1 = model(**inputs)
output2 = xlmr(**inputs)



p = 1