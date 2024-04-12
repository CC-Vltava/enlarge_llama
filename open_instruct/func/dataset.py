import datasets
from datasets import load_dataset
import torch
from langdetect import detect

def preprocess_function(examples):
    inputs = str(examples['instruction']) + " " + str(examples['input'])
    outputs = str(examples['output'])
    return {'input': inputs, 'output': outputs}

def read_data_tiger():
    print('tiger dataset')
    print('start reading tiger data')
    ds_sft = datasets.load_dataset('/data1/cchuan/dataset/tiger_data')
    print('start processing tiger data')
    return ds_sft.map(
        preprocess_function,
        num_proc=16
    )

def read_data(path):
    if 'tiger' in path:
        return read_data_tiger()
    return load_dataset('json', data_files=path, field='train')
