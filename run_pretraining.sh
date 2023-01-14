python -m torch.distributed.launch run_pretraining.py \
    --config_name=bert_large_config.json \
    --input_dir=./ \
    --output_dir=./ \
    --eval_dir=./ \
    --do_train
