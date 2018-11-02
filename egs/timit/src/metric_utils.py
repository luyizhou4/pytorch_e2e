# -*- coding: utf-8 -*-
# @Author: luyizhou4
# @Date:   2018-11-01 17:32:14
# @Function: metirc function caculating PER/CER/WER            
# @Last Modified time: 2018-11-02 15:30:07

import torch
import numpy as np
import logging


def pred_best(p):
    ret = []
    p1 = [0] + p
    for i in range(len(p)):
        c1 = p1[i]
        c2 = p1[i + 1]
        if c2 == 0 or c2 == c1:
            continue
        ret.append(c2)
    return ret

def remove_padding(l, padding_id=-1):
    ret = []
    for i in range(len(l)):
        if l[i] == padding_id:
            break
        ret.append(l[i])
    return ret

# label is done with remove_blank
# pred is got from pred_best
def levenshtein_distance(label, pred):
    n_label = len(label) + 1
    n_pred = len(pred) + 1
    if (label == pred):
        return 0
    if (len(label) == 0):
        return len(pred)
    if (len(pred) == 0):
        return len(label)

    v0 = [i for i in range(n_label)]
    v1 = [0 for i in range(n_label)]

    for i in range(len(pred)):
        v1[0] = i + 1

        for j in range(len(label)):
            cost = 0 if label[j] == pred[i] else 1
            v1[j + 1] = min(v1[j] + 1, v0[j + 1] + 1, v0[j] + cost)

        for j in range(n_label):
            v0[j] = v1[j]

    return v1[len(label)]


def CER(labels, preds, preds_lens, show_results=False, show_num=0):
    '''
        labels: torch tensors(batch_size, max_label_length) padded by -1
        preds: torch tensors shape of (max_frame_num, batchSize, alphabet_size)
        preds_len: torch tensors shape of (batch, ), contains the true lens of preds tensors
    '''
    preds = preds.transpose(0, 1).numpy()
    # (bs, max_frame_num)
    preds = np.argmax(preds, axis=2)
    # (bs, )
    preds_lens = preds_lens.numpy()
    # (bs, max_label_length)
    labels = labels.numpy()

    batch_size = preds.shape[0]

    batch_n_token = 0 # total token_num in batch
    batch_l_dist = 0 # total levenshtein_distance in batch

    for i in range(batch_size):
        l = remove_padding(labels[i])
        # note that the training seq's label maybe empty seq
        if len(l) == 0:
            continue        
        batch_n_token += len(l)
        # print("label length: "+str(len(l)))
        p = preds[i, :preds_lens[i]].tolist()
        p = pred_best(p)
        l_distance = levenshtein_distance(l, p)
        # print("l_distance: "+str(l_distance))
        batch_l_dist += l_distance

        if show_results & show_num > 0:
            show_num -= 1
            logging.info("label: " + str(l))
            logging.info("prediction: " + str(p))
            logging.info("l_distance: %d"%(l_distance))

    if show_results:
        logging.info("Batch l_distance: %d, token_num: %d, CER: %f"%(batch_l_dist, batch_n_token, 
                            1.0*batch_l_dist/batch_n_token))

    return batch_l_dist, batch_n_token
