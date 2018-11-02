#!/bin/bash
#SBATCH --job-name=timit
#SBATCH --mem=30G
#SBATCH --cpus-per-task=1


# Author:luyizhou
# Date: 2018.9.27
[ -f path.sh ] && . ./path.sh

verbose=1 #0 means Skip DEBUG/INFO messages
exp_dir="./exp/baseline_dir/timit_test"
mkdir -p ${exp_dir}
cfg_file="./configfile/debug.cfg"
cp ${cfg_file} $exp_dir/ # record the cfg file

### !!!dangerous
##########################
rm $exp_dir/train.log
#########################

touch $exp_dir/train.log
hostname | tee -a $exp_dir/train.log
# nvidia-smi | tee -a $gpu_info
# echo [    INFO]CUDA_VISIBLE_DEVICES:$CUDA_VISIBLE_DEVICES | tee -a $gpu_info

python -u ./src/train.py \
		--configfile_path ${cfg_file} \
		--verbose ${verbose} \
		--expdir ${exp_dir} \
		--logfile ${exp_dir}/train.log

