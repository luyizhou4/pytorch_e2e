# -*- coding: utf-8 -*-
# @Author: luyizhou4
# @Date:   2018-10-18 14:22:44
# @Function: create dict from ${data_root}/train/text,
#            $1 is the path to ${data_root}/train/text
#            $2 is the result dict path         
# @Last Modified time: 2018-10-18 17:19:35

import sys

dict_set = set()
def main():
    with open(sys.argv[1], 'r') as fd:
        for line in fd.readlines():
            line = line.strip()
            if line == '':
                continue
            # split by space and ignore the first id
            res_list = line.split()[1:]
            for i in range(len(res_list)):
                dict_set.add(res_list[i])
    # write dict
    with open(sys.argv[2], 'w') as fd:
        i = 0
        for symbol in sorted(dict_set):
            if i != (len(dict_set) - 1) :
                fd.write(symbol+'\n')
            else:
                fd.write(symbol)
            i += 1

if __name__ == '__main__':
    main()