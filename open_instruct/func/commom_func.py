import os
import shutil
import torch

def delete_folder(folder_path):
    """
    删除指定文件夹及其内容（如果存在）。
    
    参数：
        folder_path: 要删除的文件夹路径。
    """
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        print(f"文件夹 '{folder_path}' 已成功删除。")
    else:
        print(f"文件夹 '{folder_path}' 不存在，无需删除。")


def count_parameters(model, name=''):
    for p in model.parameters():
        print('type {}'.format(p.dtype))
        break
    tot_bf16 = sum(p.numel() for p in model.parameters() if type(p) == torch.float32)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total_params = trainable_params + frozen_params
    print('bf16 {}\ntotal paras: {}\ntrainable paras: {}\nfrozen paras: {}'.format(tot_bf16, total_params, trainable_params, frozen_params))
    return total_params, trainable_params, frozen_params
