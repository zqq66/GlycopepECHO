# GlycopepECHO

## Dependency
> python setup.py build_ext --inplace

## De novo sequencing
* Data preprocessing and peptide assignment
  
> bash inf_parallel_preprocess.sh 20 ${file_path} ${output_csv} "fragmentation mode(ethcd/scehcd)" ${output_csv} "enzyme cleavage site" ""miss cleavaged"

"enzyme cleavage site" needs to elaborate all amino acids. For example, for trypsin, you should enter "KR"

Note that by default file path also include MS/MS files in mgf format.

This will generate the input features required for model training, "preprocessed files" will be saved in the same directory as "file path"

Preprocessing can take long, the second argument of the command can be set up based on the number of threads your computer can afford for preprocessing.

* Model Inference
> PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_LAUNCH_BLOCKING=1 torchrun --standalone --nnodes=1 --nproc_per_node=5 main-inference-from-scratch.py inference.beam_size=9 test_spec_header_path=${file_path}/${output_csv} test_dataset_dir=${file_path}


## Training
* Data preprocessing
  
  > bash parallel_preprocess.sh 20 ${file_path} ${label_csv}
  
* Training

  > PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True torchrun --standalone --nnodes=1 --nproc_per_node=6 main-train.py model_path=save/ethcd.pt train_spec_header_path=${file_path}/${train_csv} val_spec_header_path=${file_path}/${val_csv} train_dataset_dir=${file_path} val_dataset_dir=${file_path}

For single GPU user
  
  > PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 main-train.py model_path=save/ethcd.pt train_spec_header_path=${file_path}/${train_csv}val_spec_header_path=${file_path}/${train_csv} train_dataset_dir=${file_path} val_dataset_dir=${file_path}

The batch size included in the configs/sample/A100_40g is tuned based on A100 GPU with 40 GB. If you are using different GPUs, please feel free to change the "bin_batch_size" to get rid of OOM error or achieve more efficient training.

