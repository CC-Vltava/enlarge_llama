import torch
import torch.nn as nn
from transformers import XLMRobertaModel, XLMRobertaConfig
import json

class para(nn.Module):
    def __init__(
            self,
            input_length,
        ):
        super(para, self).__init__()
        print('cc nb length is')
        print(input_length)
        self.ccsb = nn.Parameter(torch.sort(torch.abs(torch.rand(input_length, ))).values, requires_grad=True)
        # self.ccsb = torch.sort(self.ccsb)

    def weighted_average(self, tensors):
        # 检查输入是否合法
        assert len(tensors) == len(self.ccsb), "The number of tensors and self.ccsb must be the same."

        # 使用torch.stack将tuple中的tensor堆叠在一起
        stacked_tensors = torch.stack(tensors)

        # print(self.ccsb.device)
        # print(stacked_tensors.device)
        # 计算加权平均
        weighted_avg = torch.sum(stacked_tensors * self.ccsb.view(-1, 1, 1, 1), dim=0) / torch.sum(self.ccsb)

        return weighted_avg
    
    def forward(self, tensors):
        return self.weighted_average(tensors)

class proj(nn.Module):
    def __init__(
            self, 
            input_size,
            output_size,
            # 这里的Transformer的num_head需要和xlmr的一致
            # 可以直接使用xlmr代码中的Transformer
            mid_hidden_size=512,
            num_hidden_layers=13
        ):
        super(proj, self).__init__()

        config_path = '/data1/cchuan/data/weight/project/config.json'
        with open(config_path, "r", encoding='utf-8-sig') as f:
            config_dict = json.load(f)

        config = XLMRobertaConfig(**config_dict)

        # self.transformer = XLMRobertaModel(config)

        self.para = para(input_length=num_hidden_layers)

        self.proj1 = nn.Linear(input_size, mid_hidden_size)
        self.proj2 = nn.Linear(mid_hidden_size, output_size)
        self.proj = nn.Sequential(
            self.proj1,
            self.proj2
        )



    def forward(self, hidden_states):
        # 在这个简单的例子中，前向传播直接使用线性层
        # print(x.size())
        # x = self.transformer(inputs_embeds=hidden_states[-1]).last_hidden_state
        # x = self.para(hidden_states)
        x = self.proj(hidden_states[-1])
        return x
