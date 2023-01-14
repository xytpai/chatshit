import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Bert pretraining")

    parser.add_argument("--config_name", type=str, default=None, required=True,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--input_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain .hdf5 files  for the task.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--eval_dir", default=None, type=str, required=True,
                        help="The eval data dir. Should contain .hdf5 files for the task.")

    parser.add_argument("--eval_iter_start_samples", default=3000000, type=int,
                        help="Sample to begin performing eval.")
    parser.add_argument("--eval_iter_samples", default=16, type=int,
                        help="If set to -1, disable eval, else evaluate every eval_iter_samples during training")
    parser.add_argument("--num_eval_examples", default=10000, type=int,
                        help="number of eval examples to run eval on")

    parser.add_argument("--init_checkpoint", default=None, type=str,
                        help="The initial checkpoint to start training from.")

    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Initial learning rate (after the potential warmup period) to use.")

    parser.add_argument("--max_predictions_per_seq", default=76, type=int,
                        help="The maximum total of masked tokens in input sequence")
    parser.add_argument("--train_batch_size", default=8,
                        type=int, help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=16,
                        type=int, help="Total batch size for training.")
    parser.add_argument("--weight_decay", default=0.01,
                        type=float, help="weight decay rate.")

    parser.add_argument("--opt_lamb_beta_1", default=0.9,
                        type=float, help="LAMB beta1.")
    parser.add_argument("--opt_lamb_beta_2", default=0.999,
                        type=float, help="LAMB beta2.")
    parser.add_argument("--max_steps", default=1536, type=float,
                        help="Total number of training steps to perform.")
    parser.add_argument("--max_samples_termination", default=14000000, type=float,
                        help="Total number of training samples to run.")
    parser.add_argument("--warmup_proportion", default=0.01, type=float,
                        help="Proportion of optimizer update steps to perform linear learning rate warmup for. Typically 1/8th of steps for Phase2")
    parser.add_argument("--warmup_steps", default=0, type=float,
                        help="Number of optimizer update steps to perform linear learning rate warmup for. Typically 1/8th of steps for Phase2")
    parser.add_argument("--start_warmup_step", default=0, type=float,
                        help="Starting step for warmup. ")

    parser.add_argument("--resume_from_checkpoint", default=False, action='store_true',
                        help="Whether to resume training from checkpoint.")
    parser.add_argument('--keep_n_most_recent_checkpoints', type=int, default=20,
                        help="Number of checkpoints to keep (rolling basis).")
    parser.add_argument('--num_samples_per_checkpoint', type=int, default=500000,
                        help="Number of update steps until a model checkpoint is saved to disk.")
    parser.add_argument('--min_samples_to_start_checkpoints', type=int, default=3000000,
                        help="Number of update steps until model checkpoints start saving to disk.")
    parser.add_argument('--skip_checkpoint', default=False, action='store_true',
                        help="Whether to save checkpoints")

    parser.add_argument('--phase2', default=False, action='store_true',
                        help="Only required for checkpoint saving format")

    parser.add_argument("--do_train", default=False, action='store_true',
                        help="Whether to run training.")

    parser.add_argument('--target_mlm_accuracy', type=float, default=0.72,
                        help="Stop training after reaching this Masked-LM accuracy")
    parser.add_argument('--train_mlm_accuracy_window_size', type=int, default=0,
                        help="Average accuracy over this amount of batches before performing a stopping criterion test")

    parser.add_argument("--dense_seq_output", default=False, action='store_true',
                        help="Whether to run with optimizations.")

    # For dtype specific training
    parser.add_argument("--bf16", default=False,
                        action='store_true', help="Enale BFloat16 training")
    parser.add_argument("--fp16", default=False,
                        action='store_true', help="Enale Half training")
    parser.add_argument("--bf32", default=False,
                        action='store_true', help="Enale BFloat32 training")

    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")

    parser.add_argument("--seed", type=int, default=42,
                        help="A seed for reproducible training.")

    parser.add_argument("--profile", action="store_true",
                        help="Whether to enable profiling")
    parser.add_argument("--export_chrome_trace", action="store_true",
                        help="Exports the collected trace in Chrome JSON format.")
    parser.add_argument("--benchmark_steps", type=int, default=0,
                        help="Number of steps to run for benchmark.")

    parser.add_argument("--no_ddp", default=False,
                        action="store_true", help="Whether to use DDP.")
    parser.add_argument("--local_rank", default=0, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--world_size", default=1, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--use_gradient_as_bucket_view", default=False, action='store_true',
                        help="Turn ON gradient_as_bucket_view optimization in native DDP.")
    parser.add_argument("--dist_backend", type=str, default="ccl",
                        help="Specify distributed backend to use.")
    parser.add_argument('--dist-url', default='127.0.0.1', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-port', default='29500', type=str,
                        help='url port used to set up distributed training')
    parser.add_argument("--multi_instance", action="store_true",
                        help="Skip torch.distributed initialization to Run multiple instance independently")
    parser.add_argument("--dist_profile", action="store_true",
                        help="Whether to enable distributed timing profile")
    parser.add_argument("--no_dense_sequence_output", default=False, action='store_true',
                        help="Disable dense sequence output")

    parser.add_argument("--device", type=str,
                        default="cuda", help="backend to run")
    parser.add_argument("--amp", action="store_true",
                        help="Whether to enable autocast")
    parser.add_argument("--lamb", action="store_true")
    parser.add_argument("--adamw", action="store_true")
    parser.add_argument("--show_info", action="store_true")

    args = parser.parse_args()
    if os.environ.get("PROFILE", "OFF").upper() in ["1", "Y", "ON", "YES", "TRUE"]:
        args.profile = True
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    return args
