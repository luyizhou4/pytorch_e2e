#!/bin/bash
#SBATCH --job-name=decoding
#SBATCH --mem=10G
#SBATCH -p gpu
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node 1


# @Author: luyizhou4
# @Date:   2019-01-21 18:50:48
# @Last Modified by:   luyizhou4
# @Last Modified time: 2019-01-21 20:48:09

[ -f path.sh ] && . ./path.sh

verbose=1 #0 means Skip DEBUG/INFO messages
exp_dir="./exp/timit_exp/timit_test/decoding"
mkdir -p ${exp_dir}

cfg_file="./configfile/train.cfg"
cp ${cfg_file} $exp_dir/ # record the cfg file


echo "dec.log" > $exp_dir/dec.log
echo $exp_dir | tee -a $exp_dir/dec.log
hostname | tee -a $exp_dir/dec.log
nvidia-smi | tee -a $exp_dir/dec.log
echo CUDA_VISIBLE_DEVICES:$CUDA_VISIBLE_DEVICES | tee -a $exp_dir/dec.log

python -u ./src/predict.py \
        --configfile_path ${cfg_file} \
        --verbose ${verbose} \
        --expdir ${exp_dir} \
        --logfile ${exp_dir}/dec.log

echo "Decoding finished!"