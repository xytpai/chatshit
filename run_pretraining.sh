python -m torch.distributed.launch run_pretraining.py \
    --config_name=bert_large_config.json \
    --input_dir=/data/wiki/results/hdf5 \
    --output_dir=./ \
    --eval_dir=./ \
    --train_batch_size=2 \
    --do_train
