#!/bin/bash
#SBATCH --job-name=9
#SBATCH --mail-user=cse.cuhk.edu.hk
#SBATCH --mail-type=ALL
#SBATCH --output=output6.txt
#SBATCH --gres=gpu:1
#SBATCH -c 2


export PATH="/mnt/backup/home/xgxu/anaconda3_3/bin:$PATH"
## python main.py
## python main_test.py
## python main_rx.py --checkpoint_filepath /mnt/proj3/xgxu/comparison/DeepLPF/log_2020-12-23_20-34-41/deep_lpf_200.pth
## python main_rx.py
python main_test_rx.py
