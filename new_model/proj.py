﻿import torch
import torch.nn as nn
from transformers import XLMRobertaModel, XLMRobertaConfig
import json

class proj(nn.Module):
    def __init__(
            self, 
            input_size,
            output_size,
            # 这里的Transformer的num_head需要和xlmr的一致
            # 可以直接使用xlmr代码中的Transformer
            mid_hidden_size=512
        ):
        super(proj, self).__init__()

        config_path = '/data1/cchuan/data/weight/project/config.json'
        with open(config_path, "r", encoding='utf-8-sig') as f:
            config_dict = json.load(f)

        config = XLMRobertaConfig(**config_dict)

        self.transformer = XLMRobertaModel(config)

        self.proj1 = nn.Linear(input_size, mid_hidden_size)
        self.proj2 = nn.Linear(mid_hidden_size, output_size)
        self.proj = nn.Sequential(
            self.proj1,
            self.proj2
        )

    def forward(self, hidden_states):
        # 在这个简单的例子中，前向传播直接使用线性层
        # print(x.size())
        x = self.transformer(inputs_embeds=hidden_states[-1]).last_hidden_state
        x = self.proj(x)
        return x
