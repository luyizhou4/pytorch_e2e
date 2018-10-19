# -*- coding: utf-8 -*-
# @Author: luyizhou4
# @Date:   2018-10-18 19:29:13
# @Function: json test           
# @Last Modified time: 2018-10-19 10:27:56

import json

def main():
    json_str = {
        "sw02001-A_000098-001156": {
            "input": [
                {
                    "feat": "/mnt/lustre/sjtu/users/yzl23/work_dir/espnet_exp/espnet/exps_yzl23/swbd/dump/train_dev/deltafalse/feats.1.ark:24",
                    "name": "input1",
                    "shape": [
                        1056,
                        83
                    ]
                }
            ],
            "output": [
                {
                    "name": "target1",
                    "shape": [
                        137,
                        48
                    ],
                    "text": "hi um yeah i'd like to talk about how you dress for work and and um what do you normally what type of outfit do you normally have to wear",
                    "token": "h i <space> u m <space> y e a h <space> i ' d <space> l i k e <space> t o <space> t a l k <space> a b o u t <space> h o w <space> y o u <space> d r e s s <space> f o r <space> w o r k <space> a n d <space> a n d <space> u m <space> w h a t <space> d o <space> y o u <space> n o r m a l l y <space> w h a t <space> t y p e <space> o f <space> o u t f i t <space> d o <space> y o u <space> n o r m a l l y <space> h a v e <space> t o <space> w e a r",
                    "tokenid": "28 29 17 41 33 17 45 25 21 28 17 29 3 24 17 32 29 31 25 17 40 35 17 40 21 32 31 17 21 22 35 41 40 17 28 35 43 17 45 35 41 17 24 38 25 39 39 17 26 35 38 17 43 35 38 31 17 21 34 24 17 21 34 24 17 41 33 17 43 28 21 40 17 24 35 17 45 35 41 17 34 35 38 33 21 32 32 45 17 43 28 21 40 17 40 45 36 25 17 35 26 17 35 41 40 26 29 40 17 24 35 17 45 35 41 17 34 35 38 33 21 32 32 45 17 28 21 42 25 17 40 35 17 43 25 21 38"
                }
            ],
            "utt2spk": "2001-A"
        },
        "sw02001-A_002736-002893": {
            "input": [
                {
                    "feat": "/mnt/lustre/sjtu/users/yzl23/work_dir/espnet_exp/espnet/exps_yzl23/swbd/dump/train_dev/deltafalse/feats.1.ark:88381",
                    "name": "input1",
                    "shape": [
                        155,
                        83
                    ]
                }
            ],
            "output": [
                {
                    "name": "target1",
                    "shape": [
                        6,
                        48
                    ],
                    "text": "and is",
                    "token": "a n d <space> i s",
                    "tokenid": "21 34 24 17 29 39"
                }
            ],
            "utt2spk": "2001-A"
        } 
    } 
    with open('./data.json', 'w') as f:
        json.dump(json_str, f, indent=4, separators=(',', ': '),sort_keys=True)


if __name__ == '__main__':
    main()