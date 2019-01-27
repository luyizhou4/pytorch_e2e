# -*- coding: utf-8 -*-
# @Author: luyizhou4
# @Date:   2018-11-01 17:32:14
# @Function: metirc function caculating PER/CER/WER            
# @Last Modified time: 2019-01-27 16:20:39

import torch
import numpy as np
import math
import logging
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from numpy import newaxis

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis=2)[:, :, newaxis])
    return e_x / e_x.sum(axis=2)[:, :, newaxis]

# plot figure to show peak prob. property
def plot_peak_property(labels, preds, preds_lens):
    '''
        labels: torch tensors(batch_size, max_label_length) padded by -1
        preds: torch tensors shape of (max_frame_num, batchSize, alphabet_size)
        preds_len: torch tensors shape of (batch, ), contains the true lens of preds tensors
    '''
    preds = softmax(preds.transpose(0, 1).numpy()) # (batch_size, max_frame_num, alphabet_size) 
    # (bs, )
    preds_lens = preds_lens.numpy()
    # (bs, max_label_length)
    labels = labels.numpy()
    batch_size = preds.shape[0]
    alphabet_size = preds.shape[2]
    # print(str(preds.shape))
    # print(preds[1, 300:, :])
    # print(np.argmax(preds[1, 300:, :], axis=1))
    # np.savetxt("./data/preds.txt", (preds[19, :preds_lens[19], :]))
    # np.savetxt("./data/labels.txt", labels[19])
    for i in range(batch_size):
        l = remove_padding(labels[i])
        # note that the training seq's label maybe empty seq
        if len(l) == 0:
            continue
        plt.cla()
        plt.title(str(i) + 'Peak Output Figure')
        for j in range(alphabet_size):
            plt.plot(preds[i, :30, j])

        plt.xlabel(str(preds_lens[i]) + 'phones')
        plt.ylabel('Prob.')
        plt.savefig("./exp/timit_exp/timit_test/decoding/" + str(i) +"_peak.png")

        # max prob figure
        plt.cla()
        plt.title(str(i) + 'Max Prob Output Figure')

        plt.plot(np.max(preds[i, :, :], axis=1))

        plt.xlabel(str(preds_lens[i]) + 'phones')
        plt.ylabel('Prob.')
        plt.savefig("./exp/timit_exp/timit_test/decoding/" + str(i) +"_max_prob.png")

        # blank output figure
        plt.cla()
        plt.title(str(i) + 'blank Output Figure')
        plt.plot(preds[i, :, 0])
        plt.xlabel(str(preds_lens[i]) + 'phones')
        plt.ylabel('Prob.')
        plt.savefig("./exp/timit_exp/timit_test/decoding/" + str(i) +"_blank.png")

        # blk, max_prob, others
        plt.cla()
        plt.title(str(i) + 'Error Pattern Figure')
        plt.plot(preds[i, :40, 0], label='blank')
        plt.plot(np.max(preds[i, :40, 1:], axis=1), label='max_prob except blk')
        plt.plot(1 - np.max(preds[i, :40, 1:], axis=1) - preds[i, :40, 0], label='others')
        plt.plot()
        plt.xlabel(str(preds_lens[i]) + 'phones')
        plt.ylabel('Prob.')
        plt.savefig("./exp/timit_exp/timit_test/decoding/" + str(i) +"_error_pattern.png")

# p is a list
def pred_best(p):
    # 0: blk
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

# label after remove_blank
# pred is from pred_best
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

# best path decoding
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

# prefix search decoding, return list of output labels
def prefix_search_decoding(preds):
    '''
        preds: numpy array(true_frame_num, label_i_probs)
        Note that blk id is 0
        return:
            list of output labels
    '''
    output = []
    T, alphabet_size = preds.shape
    # initialisation
    gamma_b = [{} for t in range(T)]
    gamma_n = [{} for t in range(T)]
    for t in range(T):
        gamma_n[t][''] = 0
        gamma_b[t][''] = preds[t, 0] if t == 0 else preds[t, 0] * gamma_b[t-1][''] # TODO: should be log sum?

    prob_out = {}
    prob_prefix = {}
    prob_out[''] = gamma_b[T-1][''] # p(\phi|x)
    prob_prefix[''] = 1 - prob_out[''] # p(\phi...|x)
    # default output blk
    l_star = ''
    p_star = ''
    P = {'': prob_prefix['']} # candidate list

    # algorithm start 
    while prob_prefix[p_star] > prob_out[l_star]:  # loop while prefix prob. larger than output str prob. 
        probRemaining = prob_prefix[p_star]
        for label_k in range(1, alphabet_size):
            p = p_star + ' ' + str(label_k) if p_star != '' else str(label_k)
            gamma_n[0][p] = preds[0, label_k] if p_star == '' else 0
            gamma_b[0][p] = 0

            prefixProb = gamma_n[0][p]
            for t in range(1, T):
                # similar to CTC forward step
                newLabelProb = gamma_b[t-1][p_star] if p_star.endswith(str(label_k)) \
                                            else gamma_b[t-1][p_star] + gamma_n[t-1][p_star]

                gamma_n[t][p] = preds[t, label_k] * (newLabelProb + gamma_n[t-1][p])
                gamma_b[t][p] = preds[t, 0] * (gamma_b[t-1][p] + gamma_n[t-1][p])
                prefixProb += preds[t, label_k] * newLabelProb

            prob_out[p] = gamma_n[T-1][p] + gamma_b[T-1][p]
            prob_prefix[p] = prefixProb - prob_out[p]
            probRemaining -= prob_prefix[p]

            if prob_out[p] > prob_out[l_star]:
                l_star = p
            if prob_prefix[p] > prob_out[l_star]:
                P[p] = prob_prefix[p]
            if probRemaining <= prob_out[l_star]:
                break;

        # remove p_star from P
        if p_star in P:
            del P[p_star]
        # print('res')
        # print(l_star)
        # print(p_star)
        if P:
            p_star = max(P,key=P.get)

    return l_star

# derived from https://github.com/githubharald/CTCDecoder/blob/master/src/PrefixSearch.py
# still too slow
def heuristic_split_prefix_search_decoding(preds, thres=0.9):
    '''
        speed up prefix computation by splitting sequence into subsequences as described by Graves 
    '''
    T, alphabet_size = preds.shape

    # split sequence into 3 subsequences, splitting points should be roughly placed at 1/3 and 2/3
    splitTargets = [int(T * 1 / 4), int(T * 2 / 4), int(T * 3 / 4)]
    best = [{'target' : s, 'bestDist' : T, 'bestIdx' : s} for s in splitTargets]

    # find good splitting points (blanks above threshold, closest to uniformly divided points)
    for t in range(T):
        for b in best:
            if preds[t, 0] > thres and abs(t - b['target']) < b['bestDist']:
                b['bestDist'] = abs(t - b['target'])
                b['bestIdx'] = t
                break

    # splitting points plus begin and end of sequence
    ranges = [0] + [b['bestIdx'] for b in best] + [T]

    # do prefix search for each subsequence and concatenate results
    res = []
    for i in range(len(ranges) - 1):
        beg = ranges[i]
        end = ranges[i + 1]
        res.extend( map( int, prefix_search_decoding(preds[beg:end, :]).split() ) )
    # print(res)
    return res

# prefix search decoding
def prefix_search_CER(labels, preds, preds_lens, show_results=False, show_num=0):
    '''
        labels: torch tensors(batch_size, max_label_length) padded by -1
        preds: torch tensors shape of (max_frame_num, batchSize, alphabet_size)
        preds_len: torch tensors shape of (batch, ), contains the true lens of preds tensors
    '''
    # preds = preds.transpose(0, 1).numpy()
    preds = softmax(preds.transpose(0, 1).numpy())
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
        p = heuristic_split_prefix_search_decoding(preds[i,:preds_lens[i],:])
        # print(p)
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



def main():
    preds = np.loadtxt('./data/preds.txt')
    labels = np.loadtxt('./data/labels.txt').astype(int).tolist()
    p = np.argmax(preds[:,:], axis=1).tolist()
    print('argmax res:')
    print(p)
    l = remove_padding(labels)
    p = pred_best(p)
    print("best path decoding res:")
    print(p)
    print("label:")
    print(l)
    l_distance = levenshtein_distance(l, p)
    cer = 1.0 * l_distance / len(l)
    print('levenshtein_distance: %d, label length %d, CER: %f'%(l_distance, len(l), cer))

    # prefix search decoding
    # preds = np.array([[0.4, 0, 0.6], [0.4, 0, 0.6]])
    prefix_out = heuristic_split_prefix_search_decoding(preds)
    print("prefix path decoding res:")
    print(prefix_out)
    print("label:")
    print(l)
    l_distance = levenshtein_distance(l, prefix_out)
    cer = 1.0 * l_distance / len(l)
    print('levenshtein_distance: %d, label length %d, CER: %f'%(l_distance, len(l), cer))

if __name__ == '__main__':
    main()