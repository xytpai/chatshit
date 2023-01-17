vocab_file=$1
inputs=$2

input_files=""
for f in ${inputs}; do
  input_files=${input_files},${f}
done

python create_pretraining_data.py \
    --vocab_file=${vocab_file} \
    --input_file=${input_files} \
    --output_file=${input_files} \
    --do_lower_case \
    --max_seq_length=512 \
