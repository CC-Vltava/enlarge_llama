from new_model.model import GPT
import torch
from transformers import AutoTokenizer, LlamaForCausalLM, LlamaTokenizer, AutoModel
import torch
from accelerate import Accelerator
import os
import torch.nn as nn

path = "/data1/cchuan/tiny_llama/fix"

PATH_TO_CONVERTED_MODEL=path
PATH_TO_CONVERTED_TOKENIZER1='/data1/cchuan/data/weight/xlmr/'
PATH_TO_CONVERTED_TOKENIZER2=path


weight_path = '/data1/cchuan/new_weight/2batch_tiger/test_GPT_7B/5/pytorch_model.bin'

encode_tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER1)
decode_tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER2)

model = GPT(llama_model_path=PATH_TO_CONVERTED_MODEL)
model.proj.load_state_dict(torch.load(weight_path))

print(model.proj.para.ccsb)

num = decode_tokenizer.add_special_tokens({
    "bos_token": "<s>",
    "eos_token": "</s>",
    "unk_token": "<unk>",
    "pad_token": "<pad>",
})

model.resize_token_embeddings(len(decode_tokenizer))

while True:
    # prompt = 'can you write a story happened in the forest.'
    prompt = input("输入你的字符串\n")
    # prompt = '帮我写一个发生在森林中的童话故事'

    inputs = encode_tokenizer(
        prompt,
        return_tensors='pt'
    )

    # inputs = encode_tokenizer(prompt, return_tensors="pt")

    print('input')
    print(prompt)

    print('output')
    output1 = model.generate(inputs.input_ids, attention_mask=inputs.attention_mask, max_new_tokens=120)
    output = decode_tokenizer.batch_decode(output1, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print(output)
    
p=1
