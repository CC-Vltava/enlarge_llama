#!/usr/bin/env python
# coding=utf-8

import argparse
import logging
import math
import os
import random
import datasets
from datasets import Dataset
import torch
from functools import partial
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from new_model.model import GPT
import json
from datasets import load_dataset
from langdetect import detect

load_from_pretrained=False

import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaTokenizer,
    LlamaTokenizerFast,
    SchedulerType,
    DataCollatorForSeq2Seq,
    get_scheduler,
    GPTNeoXTokenizerFast,
    GPT2Tokenizer,
    OPTForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

logger = get_logger(__name__)

def read_data():
    data_list = []

    data_path = '/data1/cchuan/data/mllm/clean_data1.json'

    with open(data_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            data_list.append(data)
    
    print('raw data size')
    print(len(data_list[0]['train']))

    formatted_data = {
        "train": [{"prompt": item["input"], "completion": item["output"]} for item in data_list[0]['train'][: 30]],
        "test": [{"prompt": item["input"], "completion": item["output"]} for item in data_list[0]['train'][: 30]],
    }

    save_data_path = '/data1/cchuan/output_file.json'

    with open(save_data_path, 'w') as json_file:
        json.dump(formatted_data, json_file, indent=4)
    
    print('finish reading')

    return load_dataset('json', data_files=save_data_path, field='train')


def collate_fn(batch):
    # batch是一个包含单个样本的列表，每个样本是一个字典{'input': input_sequence, 'labels': target_sequence}

    # 提取输入和标签
    inputs = [item['input_ids'] for item in batch]
    labels = [item['labels'] for item in batch]
    atts = [item['attention_mask'] for item in batch]
    input2 = [item['sec_input_ids'] for item in batch]

    return {
        'input_ids': torch.stack(inputs), 
        'labels': torch.stack(labels), 
        'attention_mask': torch.stack(atts),
        'sec_input_ids':  torch.stack(input2)
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="If passed, will use LORA (low-rank parameter-efficient training) to train the model.",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=64,
        help="The rank of lora.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=float,
        default=16,
        help="The alpha parameter of lora.",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.1,
        help="The dropout rate of lora modules.",
    )
    parser.add_argument(
        "--use_flash_attn",
        action="store_true",
        help="If passed, will use flash attention to train the model.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=512,
        help="The maximum total sequence length (prompt+completion) of each training example.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--warmup_ratio", type=float, default=0, help="Ratio of total training steps used for warmup."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=None,
        help="Log the training loss and learning rate every logging_steps steps.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--low_cpu_mem_usage",
        action="store_true",
        help=(
            "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded."
            "If passed, LLM loading time and RAM consumption will be benefited."
        ),
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help=(
            "Turn on gradient checkpointing. Saves memory but slows training."
        ),
    )
    parser.add_argument(
        "--use_qlora",
        action="store_true",
        help=(
            "Use qLoRA training - main thing is initialising model in quantised form. Not compatible with deepspeed."
        ),
    )
    parser.add_argument(
        '--clip_grad_norm',
        type=float,
        default=-1,
        help='Clip gradient norm. Not compatible with deepspeed (use deepspeed config instead).',
    )
    parser.add_argument(
        '--use_8bit_optimizer',
        action='store_true',
        help='Use 8bit optimizer from bitsandbytes. Not compatible with deepspeed (use deepspeed config instead).',
    )
    args = parser.parse_args()

    # Sanity checks
    if args.dataset_name is None and args.train_file is None:
        raise ValueError("Need either a dataset name or a training file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["json", "jsonl"], "`train_file` should be a json/jsonl file."
    return args


def encode_with_prompt_completion_format(example, encode_tokenizer, decode_tokenizer, max_seq_length):
    '''
    Here we assume each example has 'prompt' and 'completion' fields.
    We concatenate prompt and completion and tokenize them together because otherwise prompt will be padded/trancated 
    and it doesn't make sense to follow directly with the completion.
    '''
    # if prompt doesn't end with space and completion doesn't start with space, add space
    # print('here!!!-------------****************')
    # print(example)

    input_ids1 = encode_tokenizer(
        example['prompt'], 
        return_tensors='pt', 
        max_length=max_seq_length, 
        truncation=True,
        padding='longest'
    )['input_ids']

    input_ids2 = decode_tokenizer(
        example['completion'] + decode_tokenizer.eos_token, 
        return_tensors='pt', 
        max_length=max_seq_length, 
        truncation=True,
        padding='longest'
    )['input_ids']

    labels = decode_tokenizer(
        example['completion'] + decode_tokenizer.eos_token, 
        return_tensors='pt', 
        max_length=max_seq_length, 
        truncation=True,
        padding='longest'
    )['input_ids']
    
    pad_length = input_ids1.shape[1]

    pad_labels = torch.ones((labels.shape[0], pad_length), dtype=labels.dtype) * -100

    labels = torch.cat([pad_labels, labels], dim=1)

    attention_mask = torch.ones_like(input_ids1)

    assert labels.shape[1] == input_ids1.shape[1] + input_ids2.shape[1], 'error!'

    def is_english(sentence):
        try:
            return detect(sentence) == 'en'
        except:
            # 如果无法确定语言，当作非英文处理
            return False
    
    return {
        'input_ids': input_ids1.flatten(),
        'labels': labels.flatten(),
        'attention_mask': attention_mask.flatten(),
        'sec_input_ids': input_ids2.flatten(),
        'language': is_english(example['prompt']) and is_english(example['completion']),
        'input': example['prompt'],
        'output': example['completion']
    }


def save_with_accelerate(accelerator, model, output_dir):
    unwrapped_model = accelerator.unwrap_model(model)
    # When doing multi-gpu training, we need to use accelerator.get_state_dict(model) to get the state_dict.
    # Otherwise, sometimes the model will be saved with only part of the parameters.
    # Also, accelerator needs to use the wrapped model to get the state_dict.
    state_dict = accelerator.get_state_dict(model)
    unwrapped_model.save_pretrained(
        output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save, state_dict=state_dict
    )

def main():
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    
    accelerator.wait_for_everyone()


    raw_datasets = read_data()

    model = GPT()

    if load_from_pretrained:
        model_weights = torch.load(os.path.join(args.output_dir, '3/pytorch_model.bin'), map_location=torch.device('cuda'))
        model.load_state_dict(model_weights)

    # model.load_model('/data1/cchuan/model_weight/11.13_GPT_7B/6/pytorch_model.bin')

    encode_tokenizer = AutoTokenizer.from_pretrained('/data1/cchuan/data/weight/xlmr/')
    decode_tokenizer = AutoTokenizer.from_pretrained('/data1/cchuan/data/weight/tiny_llama/')


    # if args.model_name_or_path:
    #     if args.use_qlora:
    #         bnb_config = BitsAndBytesConfig(
    #             load_in_4bit=True,
    #             bnb_4bit_use_double_quant=True,
    #             bnb_4bit_quant_type="nf4",
    #             bnb_4bit_compute_dtype=torch.bfloat16,
    #         )
    #         device_index = accelerator.local_process_index
    #         device_map = {"": device_index} # force data-parallel training.
    #         model = AutoModelForCausalLM.from_pretrained(
    #             args.model_name_or_path,
    #             from_tf=bool(".ckpt" in args.model_name_or_path),
    #             config=config,
    #             load_in_4bit=True,
    #             quantization_config=bnb_config,
    #             device_map=device_map,
    #             torch_dtype=torch.bfloat16,
    #             use_flash_attention_2=True if args.use_flash_attn else False,
    #         )
    #     else:
    #         model = AutoModelForCausalLM.from_pretrained(
    #             args.model_name_or_path,
    #             from_tf=bool(".ckpt" in args.model_name_or_path),
    #             config=config,
    #             low_cpu_mem_usage=args.low_cpu_mem_usage,
    #             use_flash_attention_2=True if args.use_flash_attn else False,
    #         )
    # else:
    #     logger.info("Training new model from scratch")
    #     model = AutoModelForCausalLM.from_config(config)


    decode_tokenizer.add_special_tokens({
        "bos_token": "<s>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
        "pad_token": "<pad>",
    })

    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(decode_tokenizer) > embedding_size:
        model.resize_token_embeddings(len(decode_tokenizer))


    # # no default pad token for llama!
    # # here we add all special tokens again, because the default ones are not in the special_tokens_map
    # if isinstance(tokenizer, LlamaTokenizer) or isinstance(tokenizer, LlamaTokenizerFast):
    #     num_added_tokens = tokenizer.add_special_tokens({
    #         "bos_token": "<s>",
    #         "eos_token": "</s>",
    #         "unk_token": "<unk>",
    #         "pad_token": "<pad>",
    #     })
    #     assert num_added_tokens in [0, 1], "LlamaTokenizer should only add one special token - the pad_token, or no tokens if pad token present."
    # elif isinstance(tokenizer, GPTNeoXTokenizerFast):
    #     num_added_tokens = tokenizer.add_special_tokens({
    #         "pad_token": "<pad>",
    #     })
    #     assert num_added_tokens == 1, "GPTNeoXTokenizer should only add one special token - the pad_token."
    # elif isinstance(tokenizer, GPT2Tokenizer) and isinstance(model, OPTForCausalLM):
    #     num_added_tokens = tokenizer.add_special_tokens({'unk_token': '<unk>'})

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.


    # if args.use_lora:
    #     if args.use_qlora:
    #         model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)

    #     logger.info("Initializing LORA model...")
    #     peft_config = LoraConfig(
    #         task_type=TaskType.CAUSAL_LM, 
    #         inference_mode=False, 
    #         r=args.lora_rank, 
    #         lora_alpha=args.lora_alpha, 
    #         lora_dropout=args.lora_dropout,
    #         target_modules=["q_proj", "o_proj", "v_proj", "k_proj", "gate_proj", "up_proj", "down_proj"]
    #     )
    #     model = get_peft_model(model, peft_config)
    #     model.print_trainable_parameters()

    # Preprocessing the datasets.
    encode_function = partial(
        encode_with_prompt_completion_format,
        encode_tokenizer=encode_tokenizer,
        decode_tokenizer=decode_tokenizer,
        max_seq_length=args.max_seq_length,
    )


    with accelerator.main_process_first():
        # print('The size of raw data')
        # print(len(raw_datasets['train']))
        lm_datasets = raw_datasets.map(
            encode_function,
            batched=False,
            num_proc=args.preprocessing_num_workers,
            load_from_cache_file=not args.overwrite_cache,
            remove_columns=[name for name in raw_datasets["train"].column_names if name not in ["input_ids", "labels", "attention_mask"]],
            desc="Tokenizing and reformatting instruction data",
        )
        lm_datasets.set_format(type="pt")
        lm_datasets = lm_datasets.filter(lambda example: (example['language']))
        lm_datasets = lm_datasets.filter(lambda example: (example['labels'] != -100).any())
        print('The size of clean data')
        print(len(lm_datasets['train']))
    
    train_dataset = lm_datasets['train']

    if accelerator.is_local_main_process:
        print('ccccccccc')
        print(train_dataset[0]['input'])
        print(train_dataset[0]['output'])



    # Log a few random samples from the training set:
    # print('ccccccc')
    # print(len(train_dataset))
    # print(train_dataset)
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")
        print(f"Sample {index} of the training set: {train_dataset[index]}.")

    # print('ccccccc')
    # print(len(train_dataset))
    # print(train_dataset)

    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset, 
        shuffle=True, 
        # collate_fn=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding="longest"),
        collate_fn=collate_fn,
        batch_size=args.per_device_train_batch_size
    )

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    if args.use_qlora:
        from bitsandbytes.optim import AdamW
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            optim_bits=8 if args.use_8bit_optimizer else 32,
            is_paged=True
        )
    else:
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    # Create the learning rate scheduler.
    # Note: the current accelerator.step() calls the .step() of the real scheduler for the `num_processes` times. This is because they assume 
    # the user initialize the scheduler with the entire training set. In the case of data parallel training, each process only
    # sees a subset (1/num_processes) of the training set. So each time the process needs to update the lr multiple times so that the total 
    # number of updates in the end matches the num_training_steps here.
    # Here we need to set the num_training_steps to either using the entire training set (when epochs is specified) or we need to multiply the 
    # num_training_steps by num_processes so that the total number of updates matches the num_training_steps.
    num_training_steps_for_scheduler = args.max_train_steps if overrode_max_train_steps else args.max_train_steps * accelerator.num_processes
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_training_steps=num_training_steps_for_scheduler,
        num_warmup_steps=int(num_training_steps_for_scheduler * args.warmup_ratio),
    )

    # Prepare everything with `accelerator`.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    # save_with_accelerate(accelerator, model, '/data1/cchuan/test/')

    # return

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("open_instruct", experiment_config)

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            checkpoint_path = args.resume_from_checkpoint
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[
                -1
            ]  # Sorts folders by date modified, most recent checkpoint is the last
            checkpoint_path = path
            path = os.path.basename(checkpoint_path)

        accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        accelerator.load_state(path)
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = (
                int(training_difference.replace("step_", ""))
                * args.gradient_accumulation_steps
            )
            starting_epoch = resume_step // len(train_dataloader)
            completed_steps = resume_step // args.gradient_accumulation_steps
            resume_step -= starting_epoch * len(train_dataloader)

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)

    cnt = 0
    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        total_loss = 0
        if (
            args.resume_from_checkpoint
            and epoch == starting_epoch
            and resume_step is not None
        ):
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(
                train_dataloader, resume_step
            )
        else:
            active_dataloader = train_dataloader
        for step, batch in enumerate(active_dataloader):
            with accelerator.accumulate(model):
                outputs = model(**batch)   
                loss = outputs.loss
                # We keep track of the loss at each logged step
                total_loss += loss.detach().float()
                accelerator.backward(loss)
                # clip gradient norm. don't do this with deepspeed
                if accelerator.sync_gradients and args.clip_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1
                if args.logging_steps and completed_steps % args.logging_steps == 0:
                    avg_loss = accelerator.gather(total_loss).mean().item() / args.gradient_accumulation_steps / args.logging_steps
                    logger.info(f"  Step: {completed_steps}, LR: {lr_scheduler.get_last_lr()[0]}, Loss: {avg_loss}")
                    if args.with_tracking:
                        accelerator.log(
                            {
                                "learning_rate": lr_scheduler.get_last_lr()[0],
                                "train_loss": avg_loss,
                            },
                            step=completed_steps,
                        )
                    print('loss is {}'.format(avg_loss))
                    total_loss = 0
                    
                if completed_steps >= args.max_train_steps:
                    break

        # 每个epoch记录数据
        if args.output_dir is not None:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                cnt += 1
                output_path = os.path.join(args.output_dir, str(cnt))
                accelerator.save_model(model.proj, output_path)
                accelerator.save_model(model.proj.proj, os.path.join(output_path, 'proj'))
                model.proj.transformer.save_pretrained(os.path.join(output_path, 'transformer'))
                # save_with_accelerate(accelerator, model, '/data1/cchuan/test/')


    if args.with_tracking:
        accelerator.end_training()


if __name__ == "__main__":
    main()
