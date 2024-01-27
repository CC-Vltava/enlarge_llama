from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer, BertConfig, BertModel, XLMRobertaModel
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
            llama_model_path='/data1/cchuan/data/weight/vicuna',
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
    

    def forward(
        self,
        xlmr_input_ids: torch.LongTensor = None,
        xlmr_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        llama_input_ids: torch.float = None,
        llama_attention_mask: torch.float = None,
        MSE_input_ids: torch.float = None,
        MSE_attention_mask: torch.float = None,
    ):
        # print('cc nb')
        # print(xlmr_input_ids.shape)
        # print(xlmr_attention_mask.shape)
        with torch.no_grad():
            hidden_states = self.xlmr(
                input_ids=xlmr_input_ids,
                attention_mask=xlmr_attention_mask,
                output_hidden_states=True
            ).hidden_states

        fir_input_embed = self.proj(hidden_states)

        # input_length = input_embed.shape[1]

        sec_input_embed = self.llama_model.model.embed_tokens(llama_input_ids)\
            .to(fir_input_embed.dtype)

        input_embed = torch.cat([fir_input_embed, sec_input_embed], dim=1)

        attention_mask = torch.cat([xlmr_attention_mask, llama_attention_mask], dim=1)

        with torch.no_grad():
            outputs = self.llama_model(
                inputs_embeds=input_embed,
                # 这里的attention需要修改
                attention_mask=attention_mask,
                return_dict=True,
                output_hidden_states=True,
                labels=labels
            )

            return outputs.loss
