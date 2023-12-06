from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer, BertConfig, BertModel
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
import torch

from typing import Optional
from test_model_llama.proj import proj
from transformers import StoppingCriteria, StoppingCriteriaList
from fairseq import utils


class GPT(nn.Module):
    def __init__(
            self,
            llama_model_path='/data1/cchuan/data/weight/tiny_llama/',
            # llama_model_path='/data1/cchuan/data/weight/tiny_llama/',
            xlmr_model_path='/data1/cchuan/data/weight/xlmr/',
            num_attention_heads=8,
            num_hidden_layers=2,
            mid_hidden_size=512,
        ):
        super(GPT, self).__init__()

        self.xlmr = AutoModel.from_pretrained(xlmr_model_path)
        self.llama_model = AutoModelForCausalLM.from_pretrained(llama_model_path)

        feature_size = self.xlmr.config.hidden_size

        self.proj = proj(            
            feature_size=feature_size,
            hidden_state_size=self.llama_model.config.hidden_size,
            num_attention_heads=num_attention_heads,
            num_hidden_layers=num_hidden_layers,
            mid_hidden_size=mid_hidden_size
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

    def generate(self, input_ids, attention_mask):

        features = self.xlmr(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).last_hidden_state

        llama_input_embs = self.proj(features)

        output_file_path = '/data1/cchuan/input.txt'

        # 将 Tensor 写入文本文件
        with open(output_file_path, 'w') as file:
            # 将 Tensor 转换为字符串，并写入文件
            file.write(str(llama_input_embs.tolist()))

        # generate_ids = self.llama_model.generate(
        #     inputs_embeds=llama_input_embs,
        #     max_new_tokens=30,
        # )
        generate_ids = self.model.llama_model.generate(
            inputs_embeds=llama_input_embs,
            max_new_tokens=30,
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
        

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        sec_input_ids: torch.float = None,
    ):
        
        features = self.xlmr(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).last_hidden_state

        # return features

        llama_input_ids = self.proj(features)

        print(sec_input_ids)
        sec_input_embed = self.llama_model.model.embed_tokens(sec_input_ids)\
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
        outputs = self.llama_model(
            inputs_embeds=llama_input_embed,
            # 这里的attention需要修改
            attention_mask=attention_mask,
            return_dict=True,
            # labels=labels
        )

        return outputs
    

    


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
        
