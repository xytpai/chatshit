import os
import random
import argparser
import numpy as np
import torch
import torch.nn as nn
import modeling_bert

from utils import is_main_process, format_step, get_world_size, get_rank


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


def main():
    args = argparser.parse_args()
    print('========== origin args ==========\n', args)

    random.seed(args.seed + args.local_rank)
    np.random.seed(args.seed + args.local_rank)
    torch.manual_seed(args.seed + args.local_rank)

    device, args = setup_training(args)


if __name__ == '__main__':
    main()
