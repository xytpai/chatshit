#!/bin/bash

inputs=$1

# Remove doc tag and title
python ./cleanup_file.py --data=$inputs --output_suffix='.1'

# Further clean up files
for f in ${inputs}; do
  bash clean.sh ${f}.1 ${f}.2
done

# Sentence segmentation
python ./do_sentence_segmentation.py --data=$inputs --input_suffix='.2' --output_suffix='.3'

mkdir -p ./wiki_results

# Gather into fixed size packages
python ./do_gather.py --data=$inputs --input_suffix='.3' --block_size=32 --out_dir=$2
