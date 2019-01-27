#/bin/bash
# Author:luyizhou
# Date: 2018.10.18
# Function: transform kaldi-style data to json files and prepare dict

[ -f path.sh ] && . ./path.sh

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

data_root="./data/mfcc_data/"
mapping_dict="./data/48-39.phone.map"
dict_root="./data/dict/"
mkdir -p $dict_root

stage=0

# prepare dict
if [ ${stage} -le 0 ]; then
    # transform data_root/*/text
    for data_set in train dev test; do
        cp ${data_root}/${data_set}/text ${data_root}/${data_set}/text.ori
        # 48 phone text
        cp ${data_root}/${data_set}/text ${data_root}/${data_set}/text.48phone
        # do the mapping to get the 39 phone text:
        python -u ./utils/mapping_text.py ${data_root}/${data_set}/text.48phone \
                ${mapping_dict} ${data_root}/${data_set}/text.39phone
        cp ${data_root}/${data_set}/text.39phone ${data_root}/${data_set}/text 
    done
    # create dict from data_root/train/text, the result dict is sorted
    python -u ./utils/create_dict.py ${data_root}/train/text.39phone ${dict_root}/phone_dict.txt
fi

json_dir='./data/mfcc_json/'
mkdir -p ${json_dir}
# prepare json files similar to espnet, note that for timit, the text.39phone generated above is used
if [ ${stage} -le 1 ]; then
    for data_set in train dev test; do
        python -u ./utils/create_json.py ${data_root}/${data_set} ${dict_root}/phone_dict.txt \
                ${json_dir}/${data_set}.json 
    done
fi
