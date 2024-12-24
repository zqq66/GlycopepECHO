# GlycopepECHO

## Dependency
We recommend using conda to manage all dependency. After install conda

> conda install -n GlycopepECHO requirements.yaml

> python setup.py build_ext --inplace

## De novo sequencing
> bash inference.sh 20 /home/olinked/data/test_igg igg_test.csv scehcd KR 3 0124 /home/olinked/data/test_igg/protein-IgG.fasta

Example files can be downloaded from https://drive.google.com/drive/folders/1DQ5npo7t-4NYq0x51Ru1VabCcr5NjOCA?usp=sharing
## Training
* Data preprocessing
  
  > bash parallel_preprocess.sh 20 ${file_path} ${label_csv}
  
* Training

  > PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True torchrun --standalone --nnodes=1 --nproc_per_node=6 main-train.py model_path=save/ethcd.pt train_spec_header_path=${file_path}/${train_csv} val_spec_header_path=${file_path}/${val_csv} train_dataset_dir=${file_path} val_dataset_dir=${file_path}

For single GPU user
  
  > PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 main-train.py model_path=save/ethcd.pt train_spec_header_path=${file_path}/${train_csv}val_spec_header_path=${file_path}/${train_csv} train_dataset_dir=${file_path} val_dataset_dir=${file_path}

The batch size included in the configs/sample/A100_40g is tuned based on A100 GPU with 40 GB. If you are using different GPUs, please feel free to change the "bin_batch_size" to get rid of OOM error or achieve more efficient training.

