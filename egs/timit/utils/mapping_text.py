# -*- coding: utf-8 -*-
# @Author: luyizhou4
# @Date:   2018-10-18 15:33:13
# @Function:    $1 is the path to the original text
#               $2 the mapping_dict path
#               $3 the result text path
#           将各个text中的48phone映射到39 phone的text           
# @Last Modified time: 2018-10-18 17:18:25

from __future__ import print_function
import sys

phone_mapping = {}

def main():
    initial_text = sys.argv[1]
    mapping_dict = sys.argv[2]
    output_text = sys.argv[3]

    # get mapping_dict
    with open(mapping_dict, 'r') as f:
        for line in f.readlines():
            key = line.split()[0]
            value = line.split()[1]
            phone_mapping[key] = value
    
    with open(initial_text, 'r') as fd:
        w_fd = open(output_text, 'w')
        for line in fd.readlines():
            line = line.strip()
            if line == '':
                continue
            utt_id = line.split()[0]
            phone_seq_list = line.split()[1:]

            new_phone_seq_list = [utt_id]
            for phone in phone_seq_list:
                new_phone_seq_list.append(phone_mapping[phone])
            new_text = " ".join(new_phone_seq_list)
            w_fd.write(new_text+'\n')
        w_fd.close()

if __name__ == '__main__':
    main()