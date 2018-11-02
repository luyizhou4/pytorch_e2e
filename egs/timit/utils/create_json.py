# -*- coding: utf-8 -*-
# @Author: luyizhou4
# @Date:   2018-10-18 14:23:29
# @Function: create json files from ${data_root}
#            $1 is the ${data_root}/${data_set}, ${data_set}=train/dev/test
#            $2 is the dict
#            $3 is the output json file            
# @Last Modified time: 2018-10-19 20:06:16

import sys
import json
import kaldi_io

class utterance_obj(object):
    def __init__(self, id_, input_=[], output_=[]):
        '''object that contains needed .json properties
        Args:
            id_: utterance id
            input_: list of map [{speed:, feat:, shape:}]
            output_: list of map [{text:, token:, shape:, token_id:}]
        '''
        self.id = id_
        self.input = input_
        self.output = output_

    def set_input(self, input_):
        self.input = input_
    def set_output(self, output_):
        self.output = output_    
    # check if id_
    def check_all_filled(self):
        if self.id and self.input and self.output:
            return True
        else:
            return False
    # return the obj as a map
    def get_all(self):
        map_ = {"input":self.input, "output":self.output}
        return map_

def main():
    root_dir = sys.argv[1]
    dict_path = sys.argv[2]
    json_path = sys.argv[3]

    text_path = root_dir + "/text.39phone"
    feats_scp_path = root_dir + "/feats.scp"
    # map of {id:utterance_obj}, which contains all that json files need 
    utt_dict = {}
    input_ = []
    output_ = []
    # read feats.scp and construct 'input' : list of map [{speed:, feat:, shape:}]
    # further if we use speed perturb, it may becomes [{speed:, feat:, shape:}, {speed:, feat:, shape:}, ...]
    with open(feats_scp_path, 'r') as fd:
        for line in fd.readlines():
            input_map = {}
            scp_list = line.strip().split()
            if scp_list is not None and len(scp_list) == 2:
                id_ = scp_list[0]
                # now by default the speed is 1.0, further we'll implement speed perturb
                speed = 1.0
                feat = scp_list[1]

                feat_array = kaldi_io.read_mat(feat)
                shape = list(feat_array.shape)
                # debug data
                # shape = list((-1,-1))

                input_map['speed'] = speed
                input_map['feat'] = feat
                input_map['shape'] = shape
            else:
                print("Something error happened in %s. The error line is %s. Skipped this line."%(feats_scp_path ,line))
                continue
            input_list = [input_map]
            if not utt_dict.has_key(id_):
                utt_dict[id_] = utterance_obj(id_, input_=input_list)
            else:
                print("id contradicts: %s in feats.scp, need to check."%(id_))
                exit(-1)

    # deal with the output part
    phone_list = ['<blk>']
    # read dict
    with open(dict_path, 'r') as fd:
        for line in fd.readlines():
            phone_list.append(line.strip())

    # read text file and construct 'output' : list of map [{text:, token:, shape:, token_id:}]
    with open(text_path, 'r') as fd:
        ignore_text_num = 0
        for line in fd.readlines():
            output_map = {}
            line_list = line.strip().split()
            if line_list is not None and len(line_list) != 0:
                id_ = line_list[0]
                token_list = line_list[1:]
                text_str = " ".join(token_list)
                token_str = " ".join(token_list)
                shape = [len(token_list), len(phone_list)]

                token_id_list = []
                for token in token_list:
                    token_id = str(phone_list.index(token))
                    token_id_list.append(token_id)
                output_map['text'] = text_str
                output_map['token'] = token_str
                output_map['shape'] = shape
                output_map['token_id'] = " ".join(token_id_list)
            else:
                print("Something error happened. The error line is %s. Skipped this line."%(line))
                continue
            output_list = [output_map]
            # check if this utterance is in feats.scp
            if utt_dict.has_key(id_):
                if not utt_dict[id_].output:
                    utt_dict[id_].set_output(output_list)
                else:
                    print("id contradicts: %s in text, need to check. Previous output is %s"%(id_, str(utt_dict[id_].output)))
            else:
                ignore_text_num += 1
                if ignore_text_num < 10:
                    print("There is no feats correspond to %s, ignore it."%(id_))
        if ignore_text_num > 0:
            print("The total ignored text num is %d"%(ignore_text_num))

    json_map = {}

    not_filled_num = 0
    for id_ in utt_dict:
        utt_obj = utt_dict[id_]
        if utt_obj.check_all_filled():
            json_map[utt_obj.id] = utt_obj.get_all()
        else:
            not_filled_num += 1
            if not_filled_num < 10:
                print("the id:%s is not full filled, ignore it."%(id_))

    if not_filled_num > 0:
        print("The total not filled num is %d"%(not_filled_num))


    # write json files
    with open(json_path, 'w') as fd:
        json.dump(json_map, fd, indent=4, separators=(',', ': '),sort_keys=True)

if __name__ == '__main__':
    main()