#/bin/bash
# Author:luyizhou
# Date: 2018.10.18
# Function: transform kaldi-style data to json files and prepare dict

[ -f path.sh ] && . ./path.sh

data_root="./data/data-mfcc/"

mapping_dict="./data/dict/phone_map.map"
dict_root="./data/dict/"
mkdir -p $dict_root

stage=1

# prepare dict
if [ ${stage} -le 0 ]; then
    # transform data_root/*/text
    for data_set in train dev test; do
        cp ${data_root}/${data_set}/text ${data_root}/${data_set}/text.ori
        # 48 phone text
        cp ${data_root}/${data_set}/text ${data_root}/${data_set}/text.48phone
        # rm ${data_root}/${data_set}/text
        # do the mapping to get the 39 phone text:
        python -u ./utils/mapping_text.py ${data_root}/${data_set}/text.48phone ${mapping_dict} ${data_root}/${data_set}/text.39phone
        # chmod u+x ${data_root}/${data_set}/text.39phone
        # ln -s ${data_root}/${data_set}/text.39phone ${data_root}/${data_set}/text 
    done

    # create dict from data_root/train/text, the result dict is sorted
    python -u ./utils/create_dict.py ${data_root}/train/text.39phone ${dict_root}/phone_dict.txt
fi

json_dir='./data/json/'
mkdir -p ${json_dir}
# prepare json files similar to espnet
if [ ${stage} -le 1 ]; then
    for data_set in train dev test; do
        python -u ./utils/create_json.py ${data_root}/${data_set} ${dict_root}/phone_dict.txt ${json_dir}/${data_set}.json 
    done
fi
