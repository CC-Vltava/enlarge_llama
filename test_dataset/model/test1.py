from transformers import AutoTokenizer, LlamaForCausalLM, AutoModel
# from model import GPT
import torch
# from accelerate import Accelerator
# from accelerate.logging import get_logger
# from accelerate.utils import set_seed
import torch.nn as nn
import json
from proj import proj


# config_path = '/data1/cchuan/data/weight/project/config.json'
# model_path = '/data1/cchuan/data/weight/xlmr/'

# from transformers import XLMRobertaModel, XLMRobertaConfig

# # 读取配置文件
# with open(config_path, "r", encoding='utf-8-sig') as f:
#     config_dict = json.load(f)

# # 使用配置文件创建配置对象
# config = XLMRobertaConfig(**config_dict)


# # 使用配置对象初始化模型
# toeknizer = AutoTokenizer.from_pretrained(model_path)
# model1 = AutoModel.from_pretrained(model_path)
# model2 = XLMRobertaModel(config)

# text = 'hello world, my name is CC123321'
# input1 = toeknizer(text, return_tensors='pt')
# output1 = model1(**input1).last_hidden_state
# a, b = output1.size()[: -1]
# output2 = model2(inputs_embeds=output1)
# print(output2).last_hidden_state
# print(output2.shape)
# print('ok')



import os

model = proj(768, 2048)

print(model.transformer)

save_directory='/data1/cchuan/model_weight/11.16_GPT_7B/4'

# accelerator.save_model(model, save_directory)

model_weights = torch.load(os.path.join(save_directory, 'pytorch_model.bin'), map_location=torch.device('cuda'))
model.load_state_dict(model_weights)

print(model.transformer)

print(1)






