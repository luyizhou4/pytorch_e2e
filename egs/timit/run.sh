#!/bin/bash
#SBATCH --job-name=timit
#SBATCH --mem=10G
#SBATCH -p gpu
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node 1

# Author:luyizhou
# Date: 2018.9.27

[ -f path.sh ] && . ./path.sh

verbose=1 #0 means Skip DEBUG/INFO messages
exp_dir="./exp/timit_exp/timit_test"
mkdir -p ${exp_dir}
mkdir -p ./checkpoint

cfg_file="./configfile/train.cfg"
cp ${cfg_file} $exp_dir/ # record the cfg file


echo "Train.log" > $exp_dir/train.log
echo $exp_dir | tee -a $exp_dir/train.log
hostname | tee -a $exp_dir/train.log
nvidia-smi | tee -a $exp_dir/train.log
echo CUDA_VISIBLE_DEVICES:$CUDA_VISIBLE_DEVICES | tee -a $exp_dir/train.log

python -u ./src/train.py \
        --configfile_path ${cfg_file} \
        --verbose ${verbose} \
        --expdir ${exp_dir} \
        --logfile ${exp_dir}/train.log

echo "Training finished!"

