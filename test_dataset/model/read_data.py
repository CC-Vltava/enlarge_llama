import json
from datasets import Dataset
from datasets import load_dataset
import torch
from functools import partial
from torch.utils.data import DataLoader



def enfn(example):

    return {'input_ids': example['prompt'] + example['completion'], 'labels': example['completion']}


def collate_fn(batch):
    # batch是一个包含样本的列表，每个样本是一个元组 (input_sequence, target_sequence)
    input_sequences, target_sequences = zip(*batch)

    return {"input_ids": input_sequences, "labels": target_sequences}


def read_data():
    data_list = []

    data_path = '/data1/cchuan/data/mllm/clean_data1.json'

    with open(data_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            data_list.append(data)

    formatted_data = [{"prompt": item["input"], "completion": item["output"]} for item in data_list[0]['train'][: 5]]

    dataset = Dataset.from_list(formatted_data)

    dataset_new = dataset.map(
            enfn,
            batched=False,
            num_proc=16,
            # remove_columns=[name for name in raw_datasets["train"].column_names if name not in ["input_ids", "labels", "attention_mask"]],
            desc="Tokenizing and reformatting instruction data",
        )

    # train_dataloader = DataLoader(
    #     dataset_new, 
    #     shuffle=True, 
    #     # collate_fn=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding="longest"),
    #     # collate_fn=collate_fn,
    #     batch_size=10
    # )
    return train_dataloader


dataset = read_data()

print((dataset))

1