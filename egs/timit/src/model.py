# -*- coding: utf-8 -*-
# @Author: luyizhou4
# @Date:   2018-10-24 15:24:44
# @Function:            
# @Last Modified time: 2018-11-02 14:58:04

import numpy as np
import torch
import warpctc_pytorch as warp_ctc
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
import logging

def to_cuda(m, x):
    """Function to send tensor into corresponding device

    :param torch.nn.Module m: torch module
    :param torch.Tensor x: torch tensor
    :return: torch tensor located in the same place as torch module
    :rtype: torch.Tensor
    """
    assert isinstance(m, torch.nn.Module)
    device = next(m.parameters()).device
    return x.to(device)

class CTCArch(torch.nn.Module):
    """E2E module

    :param int idim: dimension of inputs
    :param int odim: dimension of outputs
    :param namespace args: argument namespace containing options
    """

    def __init__(self, idim, hidden_dim, hidden_layers, odim, dropout_rate, label_ignore_id=-1):
        super(CTCArch, self).__init__()
        self.idim = idim
        self.odim = odim
        self.loss = None
        self.loss_fn = warp_ctc.CTCLoss(size_average=False) # normalize the loss by batch size if True
        self.ignore_id = label_ignore_id

        self.nblstm = torch.nn.LSTM(idim, hidden_dim, hidden_layers, batch_first=True,
                                    dropout=dropout_rate, bidirectional=True)
        self.l_proj = torch.nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout_layer = torch.nn.Dropout(p=dropout_rate) # dropout for the BLSTM proj layer
        self.l_output = torch.nn.Linear(hidden_dim, odim)

    def forward(self, xs_pad, ilens, ys_pad):
        '''BLSTM CTC loss

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, D)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor ys_pad: batch of padded character id sequence tensor (B, Lmax)
        :return: ctc loss value
        :rtype: torch.Tensor
        '''
        # BLSTM forward part 
        xs_pack = pack_padded_sequence(xs_pad, ilens, batch_first=True)
        # flatten_parameters() is used to compact the memory of rnn modules, 
        # so as to use data parallel in modules
        self.nblstm.flatten_parameters()
        hs, _ = self.nblstm(xs_pack)
        # hs: utt list of frame x (2*hidden_dim) (2: means bidirectional)
        hs_pad, hlens = pad_packed_sequence(hs, batch_first=True)
        # (batch_size*frame_num, feat_dim)
        projected = self.l_proj(
            hs_pad.contiguous().view(-1, hs_pad.size(2)))
        hs_pad = projected.view(hs_pad.size(0), hs_pad.size(1), -1)
        hs_pad = self.dropout_layer(hs_pad)
        
        # output layer and CTC forward part, the results of blstm is hs_pad with lens: hlens
        ys_hat = self.l_output(hs_pad)
        # expected shape of seqLength x batchSize x alphabet_size
        ys_hat = ys_hat.transpose(0, 1)
        # input true lens, tensor of (batch, )
        hlens = torch.from_numpy(np.fromiter(hlens, dtype=np.int32))

        # parse padded ys, each y is a (label_dim, ) tensor, retun list of tensor y
        ys = [y[y != self.ignore_id] for y in ys_pad]  
        # output true lens
        olens = torch.from_numpy(np.fromiter(
            (y.size(0) for y in ys), dtype=np.int32))
        # change ys_pad to one-dimensional
        ys_true = torch.cat(ys).cpu().int()  # batch x olen

        # get length info
        # logging.info(self.__class__.__name__ + ' input lengths:  ' + ''.join(str(hlens).split('\n')))
        # logging.info(self.__class__.__name__ + ' output lengths: ' + ''.join(str(olens).split('\n')))

        # get ctc loss
        self.loss = None
        self.loss = to_cuda(self, self.loss_fn(ys_hat, ys_true, hlens, olens))
        # logging.info('ctc loss:' + str(float(self.loss)))

        if self.training:
            return self.loss
        else:
            # ys_hat: shape of seqLength x batchSize x alphabet_size
            # hlens: tensor of (batch, )
            return (ys_hat, hlens, self.loss)

    @staticmethod
    def get_param_size(model):
        params = 0
        for p in model.parameters():
            tmp = 1
            for x in p.size():
                tmp *= x
            params += tmp
        return params