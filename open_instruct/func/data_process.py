import torch
from functools import partial
from langdetect import detect

def data_process(args, encode_tokenizer, decode_tokenizer, raw_dataset):
    encode_function = partial(
        encode_with_prompt_completion_format,
        encode_tokenizer=encode_tokenizer,
        decode_tokenizer=decode_tokenizer,
        max_seq_length=args.max_seq_length,
    )

    lm_datasets = raw_dataset.map(
        encode_function,
        batched=False,
        num_proc=args.preprocessing_num_workers,
        load_from_cache_file=not args.overwrite_cache,
        remove_columns=[name for name in raw_dataset["train"].column_names if name not in ["input_ids", "labels", "attention_mask"]],
        desc="Tokenizing and reformatting instruction data",
    )
    print('here1')
    print(len(lm_datasets['train']))
    lm_datasets.set_format(type="pt")
    print(lm_datasets['train'][0]['language'])
    # lm_datasets = lm_datasets.filter(lambda example: (example['language']))
    print('here2')
    print(len(lm_datasets['train']))
    lm_datasets = lm_datasets.filter(lambda example: (example['labels'] != -100).any())
    print('here3')
    print(len(lm_datasets['train']))
    print('The size of clean data')
    print(len(lm_datasets['train']))

    train_dataset = lm_datasets['train']
    
    return train_dataset



def encode_with_prompt_completion_format(example, encode_tokenizer, decode_tokenizer, max_seq_length):
    '''
    Here we assume each example has 'prompt' and 'completion' fields.
    We concatenate prompt and completion and tokenize them together because otherwise prompt will be padded/trancated 
    and it doesn't make sense to follow directly with the completion.
    '''
    # if prompt doesn't end with space and completion doesn't start with space, add space
    # print('here!!!-------------****************')
    # print(example)

    # xlmr中自己带有eos token
    encode_tokenizer.padding_side='left'
    xlmr_input = encode_tokenizer(
        example['input'], 
        return_tensors='pt', 
        max_length=max_seq_length,
        truncation=True,
        padding='max_length'
    )

    decode_tokenizer.padding_side='right'
    llama_input = decode_tokenizer(
        example['output'] + decode_tokenizer.eos_token, 
        return_tensors='pt', 
        max_length=max_seq_length, 
        truncation=True,
        padding='max_length'
    )

    labels = decode_tokenizer(
        example['output'] + decode_tokenizer.eos_token, 
        return_tensors='pt', 
        max_length=max_seq_length, 
        truncation=True,
        padding='max_length'
    )
    indices = (labels['attention_mask'] == 0)
    labels['input_ids'][indices] = -100
    labels = labels['input_ids']
    pad_labels = torch.ones((1, max_seq_length), dtype=labels.dtype) * -100
    labels = torch.cat([pad_labels, labels], dim=1)

    decode_tokenizer.padding_side='left'
    MSE_input = decode_tokenizer(
        example['input'], 
        return_tensors='pt', 
        max_length=max_seq_length, 
        truncation=True,
        padding='max_length'
    )
    MSE_input_ids = torch.cat([MSE_input['input_ids'], llama_input['input_ids']], dim=1)
    MSE_attention_mask = torch.cat([MSE_input['attention_mask'], llama_input['attention_mask']], dim=1)

    
    return {
        'xlmr_input_ids': xlmr_input['input_ids'],
        'xlmr_attention_mask': xlmr_input['attention_mask'],
        'labels': labels,
        'llama_input_ids': llama_input['input_ids'],
        'llama_attention_mask': llama_input['attention_mask'],
        'language': True,
        'input': example['input'],
        'output': example['output']
    }

def collate_fn(batch):
    # batch是一个包含单个样本的列表，每个样本是一个字典{'input': input_sequence, 'labels': target_sequence}

    # 提取输入和标签
    xlmr_input_ids = [item['xlmr_input_ids'] for item in batch]
    xlmr_attention_mask = [item['xlmr_attention_mask'] for item in batch]
    llama_input_ids = [item['llama_input_ids'] for item in batch]
    llama_attention_mask = [item['llama_attention_mask'] for item in batch]
    labels = [item['labels'] for item in batch]

    return {
        'xlmr_input_ids': torch.cat(xlmr_input_ids, dim=0), 
        'xlmr_attention_mask': torch.cat(xlmr_attention_mask, dim=0),
        'llama_input_ids':  torch.cat(llama_input_ids, dim=0),
        'llama_attention_mask': torch.cat(llama_attention_mask, dim=0),
        'labels': torch.cat(labels, dim=0)
    }

