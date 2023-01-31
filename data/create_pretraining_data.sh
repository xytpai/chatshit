input_files=$1

python create_pretraining_data.py \
    --vocab_file=${VOCAB_FILE} \
    --input_file=${input_files} \
    --output_file=${input_files} \
    --do_lower_case \
    --max_seq_length=512 \
