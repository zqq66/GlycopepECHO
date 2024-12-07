#!/bin/bash
threads=$1
file_path=$2
output_csv=$3
fragmentation=$4
enzyme=$5
miss_cleavaged=$6
mass_cand=$7
fasta=$8
# isotope_shift=$8
# preprocessing
bash inf_parallel_preprocess.sh ${threads} ${file_path} ${file_path}/${output_csv} ${fragmentation} ${fasta} ${enzyme} ${miss_cleavaged}
cat ${file_path}/${output_csv}*_all_pep_mass.csv > ${file_path}/${output_csv}
# model inference
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python main-inference-from-scratch.py inference.beam_size=9 test_spec_header_path=${file_path}/${output_csv} test_dataset_dir=${file_path} inference.mass_cand=${mass_cand}
# Rescoring
python3 rescoring.py ${file_path}/GlycopepECHO-raw-output.csv ${file_path}