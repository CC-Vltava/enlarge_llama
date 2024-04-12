from transformers import AutoTokenizer, LlamaForCausalLM
import torch

path1 = '/data1/cchuan/data/weight/tiny_llama/'
path2 = '/data/ghchen/models/llama2'

model = LlamaForCausalLM.from_pretrained(path1)
tokenizer = AutoTokenizer.from_pretrained(path1)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

num = tokenizer.add_special_tokens({
    "bos_token": "<s>",
    "eos_token": "</s>",
    "unk_token": "<unk>",
    "pad_token": "<pad>",
})
# print('parameters number')
# print(count_parameters(model))
model.resize_token_embeddings(len(tokenizer))
# print(count_parameters(model))

# prompt = "Hi doctor,I have a sudden loss of erection. I got weakness in erection due to aggressive masturbation. I cannot maintain it, and my skin also became red. Is the problem due to a venous leak?"
prompt = 'translate to English：我喜欢吃草莓。'

inputs = tokenizer(prompt, return_tensors="pt")

# Generate
generate_ids = model.generate(inputs.input_ids, max_length=60)
output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        # "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
print('input: ')
print(prompt)
print('output: ')
print(output)

