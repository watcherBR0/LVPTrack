# LVPTrack
The official implementation for the AAAI 2025 paper [_LVPTrack: High Performance Domain Adaptive UAV Tracking with Label
Aligned Visual Prompt Tuning_]

## Install the environment

Create and activate a conda environment:
```
conda create -n lvptrack python=3.8
conda activate lvptrack
conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=10.2 -c pytorch
```
Then install the required packages:
```
pip install -r requirements.txt
```

## Set project paths

Run the following command to set paths for this project
```
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir ./output
```
After running this command, you can also modify paths by editing these two files
```
lib/train/admin/local.py  # paths about training
lib/test/evaluation/local.py  # paths about testing
```

## Dataset Preparation

Put the tracking datasets in ./data. It should look like this:
```
${PROJECT_ROOT}
 -- data
     -- lasot
         |-- airplane
         |-- basketball
         |-- bear
         ...
     -- got10k
         |-- test
         |-- train
         |-- val
     -- coco
         |-- annotations
         |-- images
     -- trackingnet
         |-- TRAIN_0
         |-- TRAIN_1
         ...
         |-- TRAIN_11
         |-- TEST
     -- got10k_dark
         |-- test
         |-- train
         |-- val   
     -- got10k_haze
         |-- test
         |-- train
         |-- val         
``` 
The synthetic datasets are available in [BaiduNetdisk](https://pan.baidu.com/s/1sEn0E3-Kt1X5KZYYovIYYA?pwd=es5c) and [huggingface](https://huggingface.co/datasets/WatcherBrR0/synthetic_datasets)

## Training

Download our pre-trained foundation model in [BaiduNetdisk](https://pan.baidu.com/s/1TbGy4M5XEsWPIAJaWE40vg?pwd=v85j) or [Google Drive](https://drive.google.com/drive/folders/1J-GZJTwopSkUgMJHDNX_Q_-4tdbkPumv?usp=sharing) which is based on our backbone and put it under  `$PROJECT_ROOT$/pretrained_models`. Please place the initial pseudo-labels(download form [BaiduNetdisk](https://pan.baidu.com/s/1tlvE55qxGSYQvIfAaPTSNw?pwd=2rgf) or [Google Drive](https://drive.google.com/file/d/11TEFNOeKoj4efP_5T4_1rC6Lbp8VaT8k/view?usp=sharing)) in the output folder (e.g., `./output/pseudo_label`), and execute the following command:

```
python tracking/train.py --script lightUAV --config vit_256_ep300_dark --save_dir ./output --mode multiple --nproc_per_node 4  --use_wandb 0
```
Replace `--config` with the desired model config under `experiments/lightUAV`. We use [wandb](https://github.com/wandb/client) to record detailed training logs, in case you don't want to use wandb, set `--use_wandb 0`. 

## Evaluation
Use your own training weights or ours in [BaiduNetdisk](https://pan.baidu.com/s/1WrokvrzZbljrcuMOfPEvEw?pwd=xiwg) or [Google Drive](https://drive.google.com/drive/folders/1mUZiZsFh8LHhmY0TqBjAaCiMvP8PhRy-?usp=sharing) under `$PROJECT_ROOT$/output/checkpoints/train/lightUAV`.  

Change the corresponding values of `lib/test/evaluation/local.py` to the actual benchmark saving paths

Testing examples in different domains:
- DTB70 for darkness or other off-line evaluated benchmarks (modify `--dataset` correspondingly)
```
python tracking/test.py lightUAV vit_256_ep300_dark --dataset dtb70_dark --runid 0001 --ep 300 --save_dir output
python tracking/analysis_results.py # need to modify tracker configs and names
```
- GOT10K-haze(our synthetic dataset)
```
python tracking/test.py lightUAV vit_256_ep300_haze --dataset got10k_haze --runid 0001 --ep 300 --save_dir output
python lib/test/utils/transform_got10k.py # need to modify tracker configs and names
```

## Acknowledgement
Our code is built upon [LiteTrack](https://github.com/TsingWei/LiteTrack). Also grateful for PyTracking.
