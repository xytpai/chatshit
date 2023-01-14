import os
import random
import argparser
import numpy as np
import torch
import torch.nn as nn
import modeling_bert

from utils import is_main_process, format_step, get_world_size, get_rank
from schedulers import PolyWarmUpScheduler


def setup_training(args):

    if args.device == 'cuda':

        assert (torch.cuda.is_available())

        if args.local_rank == -1:
            device = torch.device("cuda", 0)
            args.n_gpu = 1
            args.allreduce_post_accumulation = False
            args.allreduce_post_accumulation_fp16 = False
        else:
            torch.cuda.set_device(args.local_rank)
            device = torch.device("cuda", args.local_rank)
            torch.distributed.init_process_group(
                backend='nccl', init_method='env://')
            args.n_gpu = 1

        print("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
            device, args.n_gpu, bool(args.local_rank != -1), args.fp16))

        if args.gradient_accumulation_steps < 1:
            raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                args.gradient_accumulation_steps))
        if args.train_batch_size % args.gradient_accumulation_steps != 0:
            raise ValueError("Invalid gradient_accumulation_steps parameter: {}, batch size {} should be divisible".format(
                args.gradient_accumulation_steps, args.train_batch_size))

        args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

        if not args.do_train:
            raise ValueError(" `do_train`  must be True.")

        if not args.resume_from_checkpoint and os.path.exists(args.output_dir) and (
                os.listdir(args.output_dir) and any([i.startswith('ckpt') for i in os.listdir(args.output_dir)])):
            raise ValueError(
                "Output directory ({}) already exists and is not empty.".format(args.output_dir))

        if (not args.resume_from_checkpoint or not os.path.exists(args.output_dir)) and is_main_process():
            os.makedirs(args.output_dir, exist_ok=True)

        return device, args

    else:
        raise NotImplementedError


def prepare_model_and_optimizer(args, device, sequence_output_is_dense):
    config = modeling_bert.BertConfig.from_json_file('bert_large_config.json')

    # Padding for divisibility by 8
    if config.vocab_size % 8 != 0:
        config.vocab_size += 8 - (config.vocab_size % 8)

    model = modeling_bert.BertForPreTraining(
        config, sequence_output_is_dense=sequence_output_is_dense)

    checkpoint = None
    if not args.resume_from_checkpoint:
        global_step = 0
    else:
        if args.resume_step == -1 and not args.init_checkpoint:
            model_names = [f for f in os.listdir(
                args.output_dir) if f.endswith(".pt")]
            args.resume_step = max(
                [int(x.split('.pt')[0].split('_')[1].strip()) for x in model_names])
        global_step = args.resume_step if not args.init_checkpoint else 0
        if not args.init_checkpoint:
            checkpoint = torch.load(os.path.join(
                args.output_dir, "ckpt_{}.pt".format(global_step)), map_location=device)
        else:
            checkpoint = torch.load(args.init_checkpoint, map_location=device)

        model.load_state_dict(checkpoint['model'], strict=False)

        if args.phase2 and not args.init_checkpoint:
            global_step -= args.phase1_end_step
        if args.init_checkpoint:
            args.resume_step = 0
        if is_main_process():
            print("resume step from ", args.resume_step)

    model = model.to(device)
    if args.fp16 and args.allreduce_post_accumulation_fp16:
        model = model.half()
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta', 'LayerNorm']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(
            nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters,
                                  lr=args.learning_rate)
    lr_scheduler = PolyWarmUpScheduler(optimizer,
                                       warmup=args.warmup_proportion,
                                       total_steps=args.max_steps,
                                       base_lr=args.learning_rate,
                                       device=device)


def main():
    args = argparser.parse_args()
    print('========== origin args ==========\n', args)

    random.seed(args.seed + args.local_rank)
    np.random.seed(args.seed + args.local_rank)
    torch.manual_seed(args.seed + args.local_rank)

    device, args = setup_training(args)
    prepare_model_and_optimizer(
        args, device, sequence_output_is_dense=not args.no_dense_sequence_output)


if __name__ == '__main__':
    main()
