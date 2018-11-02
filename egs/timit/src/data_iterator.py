# -*- coding: utf-8 -*-
# @Author: luyizhou4
# @Date:   2018-10-24 16:01:47
# @Function:            
# @Last Modified time: 2018-11-02 12:45:30

import json
import random
import kaldi_io
import logging
import sys
import torch
import numpy as np

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler

class JsonDataset(Dataset):
    """json_file dataset."""

    def __init__(self, json_path, dataset_type, sorted_type='random', delta_feats_num=2):
        """
        Args:
            json_file (string): Path to the json file.
            dataset_type (string): train, dev, test 
            sorted_type (string): 'random', 'ascending', 'descending'. By default, we shuffle the dataset
            delta_feats_num (int): 0 means no delta feats, >0 will add #num delta_feats
        """
        if not isinstance(delta_feats_num, int) or isinstance(delta_feats_num, bool) or \
                delta_feats_num <= -1:
            raise ValueError("delta_feats_num should be a positive integeral value or zero, "
                             "but got delta_feats_num={}".format(batch_size))

        with open(json_path, 'rb') as f:
            data = json.load(f)
        self.utt_num = len(data)
        logging.info("#{} dataset total utts num: {}".format(dataset_type, self.utt_num))

        data_list = data.items() # list of (utt_id, json_infos)
        self.delta_feats_num = delta_feats_num
        self.idim = int(data_list[0][1]['input'][0]['shape'][1]) * (delta_feats_num + 1)
        self.odim = int(data_list[0][1]['output'][0]['shape'][1])
        logging.info('# input dims : ' + str(self.idim))
        logging.info('# output dims: ' + str(self.odim)) 

        if sorted_type == 'random':
            random.shuffle(data_list)
        elif sorted_type == 'ascending':
            # in lambda: data[0] is the utt_id, data[1] contains input & output infos
            data_list = sorted(data_list, key=lambda data: int(
                data[1]['input'][0]['shape'][0]), reverse=False)
        elif sorted_type == 'descending':
            data_list = sorted(data_list, key=lambda data: int(
                data[1]['input'][0]['shape'][0]), reverse=True)
        else:
            raise ValueError("sorted_type must be \'random\', \'ascending\', \'descending\', "
                                    "but got %s"%(sorted_type))
        
        self.data_list = data_list

    def __len__(self):
        return self.utt_num

    # return utt_id(string), utt_feats(np.array), transcript_ids(integer list)
    def __getitem__(self, idx):
        utt_infos = self.data_list[idx]
        utt_id = utt_infos[0]
        utt_feats = kaldi_io.read_mat(utt_infos[1]['input'][0]['feat'])
        # add delta feats
        if self.delta_feats_num > 0:
            utt_feats = [utt_feats]
            for i in range(self.delta_feats_num):
                delta_feats = self.delta(utt_feats[i],N=2)
                utt_feats.append(delta_feats)
            utt_feats = np.concatenate(utt_feats, axis=1)
        transcript_ids = map(int, utt_infos[1]['output'][0]['token_id'].strip().split())
        return (utt_id, utt_feats, transcript_ids)

    @staticmethod
    def delta(feat, N=2):
        """Compute delta features from a feature vector sequence.
        :param feat: A numpy array of size (NUMFRAMES by number of features) containing features. Each row holds 1 feature vector.
        :param N: For each frame, calculate delta features based on preceding and following N frames
        :returns: A numpy array of size (NUMFRAMES by number of features) containing delta features. Each row holds 1 delta feature vector.
        """
        if N < 1:
            raise ValueError('N must be an integer >= 1')
        NUMFRAMES = len(feat)
        denominator = 2 * sum([i**2 for i in range(1, N+1)])
        delta_feat = np.empty_like(feat)
        padded = np.pad(feat, ((N, N), (0, 0)), mode='edge')   # padded version of feat
        for t in range(NUMFRAMES):
            delta_feat[t] = np.dot(np.arange(-N, N+1), padded[t : t+2*N+1]) / denominator   # [t : t+2*N+1] == [(N+t)-N : (N+t)+N+1]
        return delta_feat

class SequentialSampler(Sampler):
    r"""Samples elements sequentially, always in the same order.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        return iter(range(len(self.data_source)))

    def __len__(self):
        return len(self.data_source)

class SimpleBatchSampler(Sampler):
    ''' 
        Simple batch sampler 
        Just feed each fixed-length-batch in order
    '''
    def __init__(self, sampler, batch_size, drop_last):

        if not isinstance(sampler, Sampler):
            raise ValueError("sampler should be an instance of "
                             "torch.utils.data.Sampler, but got sampler={}"
                             .format(sampler))
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integeral value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))

        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size

# construct the batch according to the index from SimpleBatchSampler
def _collate_fn(batch):
    '''
        input: batch is a list of tuple:
                    (utt_id(string), utt_feats(np.array), transcript_ids(integer list)) 
        return a batch to train(note that the batch should be in descending order with respect to the frame num)
    '''

    batch_size = len(batch)
    # sorting batch in descending order with respect to the frame num
    batch = sorted(batch, key=lambda sample: sample[1].shape[0], reverse=True)
    # longest input_frame_num sample
    longest_frame_sample = batch[0]
    max_frame_num = longest_frame_sample[1].shape[0]
    feats_dim = longest_frame_sample[1].shape[1]
    # padding zeros in input tensor
    np_inputs = np.zeros((batch_size, max_frame_num, feats_dim), dtype=np.float32)
    inputs_raw_frame_num = torch.IntTensor(batch_size)
    # padding -1 in output tensor
    longest_label_sample = max(batch, key=lambda sample: len(sample[2]))
    max_target_length = len(longest_label_sample[2])
    targets = -np.ones((batch_size, max_target_length), dtype=np.int32)
    targets_raw_length = torch.IntTensor(batch_size)
    utt_ids = []

    for i in range(batch_size):
        sample = batch[i]
        sample_raw_id = sample[0]
        sample_raw_input_nparray = sample[1]
        sample_raw_target_list = sample[2]

        inputs_raw_frame_num[i] = sample_raw_input_nparray.shape[0]
        np_inputs[i,:inputs_raw_frame_num[i],:] = sample_raw_input_nparray[:,:]
        targets_raw_length[i] = len(sample_raw_target_list)
        targets[i, :targets_raw_length[i]] = sample_raw_target_list[:]

        utt_ids.append(sample_raw_id)

    inputs = torch.from_numpy(np_inputs)
    targets = torch.from_numpy(targets)

    return inputs, targets, inputs_raw_frame_num, targets_raw_length, utt_ids


class E2EDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        """
        Creates a data loader for AudioDatasets.
        """
        super(E2EDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn
