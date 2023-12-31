﻿from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer, BertConfig, BertModel, XLMRobertaModel
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
import torch
import torch.nn.functional as F

from typing import Optional
from new_model.proj import proj
from transformers import StoppingCriteria, StoppingCriteriaList
from fairseq import utils


class GPT(nn.Module):
    def __init__(
            self,
            llama_model_path='/home/cchuan/Project/qlora/tiny_llama/',
            # llama_model_path='/data1/cchuan/data/weight/tiny_llama/',
            xlmr_model_path='/data1/cchuan/data/weight/xlmr/',
            mid_hidden_size=512,
        ):
        super(GPT, self).__init__()

        self.xlmr = AutoModel.from_pretrained(xlmr_model_path)
        self.llama_model = AutoModelForCausalLM.from_pretrained(llama_model_path)

        feature_size = self.xlmr.config.hidden_size

        self.proj = proj(
            input_size=feature_size,
            output_size=self.llama_model.config.hidden_size,
            mid_hidden_size=mid_hidden_size,
            num_hidden_layers=len(self.xlmr.encoder.layer) + 1
        )

        for param in self.xlmr.parameters():
            param.requires_grad = False
        for param in self.llama_model.parameters():
            param.requires_grad = False

        # 不确定
        # stop_words_ids = [torch.tensor([835]).to(self.device),
        #                   torch.tensor([2277, 29937]).to(self.device)]  # '###' can be encoded in two different ways.
        # self.stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])


    def get_input_embeddings(self):
        return self.llama_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.llama_model.set_input_embeddings(value)
        
    def resize_token_embeddings(self, size):
        self.llama_model.resize_token_embeddings(size)

    def generate(self, input_ids, attention_mask, max_new_tokens=30):

        hidden_states = self.xlmr(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        ).hidden_states

        llama_input_embs = self.proj(hidden_states)

        # output_file_path = '/data1/cchuan/input.txt'

        # # 将 Tensor 写入文本文件
        # with open(output_file_path, 'w') as file:
        #     # 将 Tensor 转换为字符串，并写入文件
        #     file.write(str(llama_input_embs.tolist()))

        # # generate_ids = self.llama_model.generate(
        # #     inputs_embeds=llama_input_embs,
        # #     max_new_tokens=30,
        # # )

        generate_ids = self.llama_model.generate(
            inputs_embeds=llama_input_embs,
            max_new_tokens=max_new_tokens,
            # stopping_criteria=self.stopping_criteria,
            num_beams=1,
            do_sample=True,
            min_length=1,
            top_p=0.9,
            repetition_penalty=1.0,
            length_penalty=1,
            temperature=1.0,
        )

        return generate_ids
        

    def calc_MSE_loss(self, model, input, sec_input, label):
        final_input = torch.cat([input, sec_input], dim=1)
        
        shape = final_input.shape
        input_length = input.shape[1]
        label_length = sec_input.shape[1]

        attention_mask = torch.zeros([shape[0], shape[1]])
        attention_mask[:, :input_length] = 1
        # print('final input {}'.format(final_input.shape))
        output = torch.stack(
            model(
                input_ids=final_input,
                attention_mask=attention_mask,
                return_dict=True,
                output_hidden_states=True
            ).hidden_states
        )

        # print('ccccccc')
        # print(output.shape, label.shape)
        # print(output[:, :, -label_length: ].shape, label[:, :, -label_length: ].shape)

        mse_loss = F.mse_loss(output[-1, :, -label_length: ], label[-1, :, -label_length: ])

        return mse_loss


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        sec_input_ids: torch.float = None,
        llama_input: torch.float = None,
    ):
        
        hidden_states = self.xlmr(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        ).hidden_states

        input_embed = self.proj(hidden_states)
        input_length = input_embed.shape[1]

        sec_input_embed = self.llama_model.model.embed_tokens(sec_input_ids)\
            .to(input_embed.dtype)

        llama_input_embed = torch.cat([input_embed, sec_input_embed], dim=1)

        shape = llama_input_embed.shape

        attention_mask = torch.zeros([shape[0], shape[1]])
        attention_mask[:, :input_length] = 1

        outputs = self.llama_model(
            inputs_embeds=llama_input_embed,
            # 这里的attention需要修改
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True,
            labels=labels
        )

        # print('here!!!')
        # print('llama_input {}'.format(llama_input.shape))
        # print('input {}'.format(input_embed.shape))
        # print('sec_input {}'.format(sec_input_ids.shape))
        # print('label {}'.format(labels.shape))
        # print('output {}'.format(outputs.hidden_states[-1].shape))
        MSE_loss = self.calc_MSE_loss(self.llama_model, llama_input, sec_input_ids, torch.stack(outputs.hidden_states))

        my_lambda = 4
        # print('compare')
        # print(outputs.loss, MSE_loss)

        return outputs.loss + my_lambda * MSE_loss
    

    


# model = GPT()

# # print('hello')
# input = [
#     'heallo',
#     '123 123 12 3123 123 123 123 '
# ]
# labels = [
#     '1jbjj ijr ',
#     '32kfjhu howihr ohdough ewruh weiuh 23o4iruh 324oiu'
# ]
# encode_tokenizer = AutoTokenizer.from_pretrained('/data1/cchuan/data/weight/xlmr/')
# decode_tokenizer = AutoTokenizer.from_pretrained('/data1/cchuan/data/weight/tiny_llama/')

# max_seq_length=256

# intput_data = encode_tokenizer(
#     labels, 
#     return_tensors='pt', 
#     max_length=max_seq_length, 
#     truncation=True,
#     padding="max_length",
# )

# num_added_tokens = decode_tokenizer.add_special_tokens({
#     "bos_token": "<s>",
#     "eos_token": "</s>",
#     "unk_token": "<unk>",
#     "pad_token": "<pad>",
# })

# embedding_size = model.get_input_embeddings().weight.shape[0]
# if len(decode_tokenizer) > embedding_size:
#     model.llama_model.resize_token_embeddings(len(decode_tokenizer))

# labels_data = decode_tokenizer(
#     labels, 
#     return_tensors='pt', 
#     max_length=max_seq_length, 
#     truncation=True,
#     padding="max_length"
# )

# outputs = model(intput_data['input_ids'], intput_data['attention_mask'], labels_data['input_ids'])

# print(outputs.loss)
        
