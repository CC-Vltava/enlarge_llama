from new_model.model import GPT
import torch
from transformers import AutoTokenizer, LlamaForCausalLM, LlamaTokenizer, AutoModel
import torch
from accelerate import Accelerator
import os
import torch.nn as nn


PATH_TO_CONVERTED_MODEL="/data1/cchuan/data/weight/tiny_llama"
PATH_TO_CONVERTED_TOKENIZER='/data1/cchuan/data/weight/tiny_llama'

weight_path = '/data1/cchuan/model_weight/11.17_GPT_7B/2/pytorch_model.bin'

tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

model = GPT()
model.proj.load_state_dict(torch.load(weight_path))


prompt = "I like cats, but I do not like dogs!"

num = tokenizer.add_special_tokens({
    "bos_token": "<s>",
    "eos_token": "</s>",
    "unk_token": "<unk>",
    "pad_token": "<pad>",
})

model.resize_token_embeddings(len(tokenizer))

inputs = tokenizer(
    prompt,
    return_tensors='pt',
    max_length=30, 
    truncation=True,
    padding='longest',
)


print('output')
output1 = model.generate(inputs.input_ids, attention_mask=inputs.attention_mask)
output = tokenizer.batch_decode(output1, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(output)

p=1