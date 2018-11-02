# -*- coding: utf-8 -*-
# @Author: luyizhou4
# @Date:   2018-10-16
# @Function: parse the configfile, set default settings and start training
       
import os
import sys
import logging
import ConfigParser
import argparse
import torch
import numpy as np
import random
import json
import time
from torch.utils.data import DataLoader

from data_iterator import JsonDataset, SimpleBatchSampler, SequentialSampler, E2EDataLoader
from model import CTCArch
from metric_utils import CER

def init_weights(m):
    

def asr_train(args):
    train_json_path = args.cfg.get('data', 'train_json')
    dev_json_path = args.cfg.get('data', 'dev_json')
    test_json_path = args.cfg.get('data', 'test_json')
    batch_size = args.cfg.getint('data', 'batch_size')
    delta_feats_num = args.cfg.getint('data', 'delta_num')
    num_workers = args.cfg.getint('data', 'num_workers')
    
    # prepare data iterator for train dataset
    train_dataset = JsonDataset(train_json_path, dataset_type='train', sorted_type='ascending', delta_feats_num=delta_feats_num)
    train_sampler = SimpleBatchSampler(sampler=SequentialSampler(train_dataset), 
                                    batch_size=batch_size,
                                    drop_last=False)
    train_data_loader = E2EDataLoader(dataset=train_dataset, 
                                    batch_sampler=train_sampler,
                                    num_workers=num_workers)
    # prepare data iterator for dev set
    dev_dataset = JsonDataset(dev_json_path, dataset_type='dev', sorted_type='random', delta_feats_num=delta_feats_num)
    dev_sampler = SimpleBatchSampler(sampler=SequentialSampler(dev_dataset), 
                                    batch_size=batch_size,
                                    drop_last=False)
    dev_data_loader = E2EDataLoader(dataset=dev_dataset, 
                                    batch_sampler=dev_sampler,
                                    num_workers=num_workers)
    # test data set 
    test_dataset = JsonDataset(test_json_path, dataset_type='test', sorted_type='random', delta_feats_num=delta_feats_num)
    test_sampler = SimpleBatchSampler(sampler=SequentialSampler(test_dataset), 
                                    batch_size=batch_size,
                                    drop_last=False)
    test_data_loader = E2EDataLoader(dataset=test_dataset, 
                                    batch_sampler=test_sampler,
                                    num_workers=num_workers)

    # get input and output dimension info
    idim = train_dataset.idim 
    odim = train_dataset.odim
    hidden_dim = args.cfg.getint('model', 'hidden_dim')
    hidden_layer = args.cfg.getint('model', 'hidden_layer')
    dropout_rate = args.cfg.getfloat('model', 'dropout_rate')

    # construct models
    model = CTCArch(idim= idim, 
                    hidden_dim=hidden_dim, 
                    hidden_layers=hidden_layer, 
                    odim=odim, 
                    dropout_rate=dropout_rate, 
                    label_ignore_id=-1)
    # optimizer type
    optimizer_type = args.cfg.get('optimizer', 'optimizer_type')
    if optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.cfg.getfloat('optimizer', 'lr'),
                            momentum=args.cfg.getfloat('optimizer', 'momentum'), 
                            nesterov=args.cfg.getboolean('optimizer', 'nesterov'))
    elif optimizer_type == 'Adadelta':
        optimizer = torch.optim.Adadelta(model.parameters())
    elif optimizer_type == 'adam':
        optimizer = torch.optim.Adam(model.parameters, lr=args.cfg.getfloat('optimizer', 'lr'))

    # set torch device
    device = torch.device("cuda" if args.cfg.getint('common', 'ngpu') > 0 else "cpu")
    model = model.to(device)

    # start training
    logging.info(model)
    logging.info("Number of parameters: %d" % CTCArch.get_param_size(model))
    for epoch_i in range(args.cfg.getint('train', 'epoch_num')):
        # training part
        train_start_time = time.time()
        model.train()
        train_loss = 0.0
        for i, (data) in enumerate(train_data_loader):
            if i == len(train_sampler):
                break
            inputs, targets, inputs_raw_length, targets_raw_length, utt_ids = data
            loss = model(inputs, inputs_raw_length, targets)
            train_loss_value = loss.item()
            train_loss += 1.0 * train_loss_value / train_dataset.__len__()
            # compute gradient
            optimizer.zero_grad()
            loss.backward()
            # SGD step
            optimizer.step()
        logging.info("epoch %d completed in %.2f s"%(epoch_i, time.time()-train_start_time))
        logging.info("epoch %d: train loss: %f"%(epoch_i, train_loss))
        print("epoch %d: train loss: %f"%(epoch_i, train_loss))

        # dev part
        model.eval()
        dev_loss = 0.0
        dev_total_l_dist = 0
        dev_total_n_token = 0
        dev_show_every = args.cfg.getint('common', 'dev_show_every')
        dev_show_num = args.cfg.getint('common', 'dev_show_num')
        show_results = False
        with torch.no_grad():
            for i, (data) in enumerate(dev_data_loader):
                if i == len(dev_sampler):
                    break
                show_results = True if i % dev_show_every == 0 else False
                inputs, targets, inputs_raw_length, targets_raw_length, utt_ids = data
                ys_hat, lens, loss = model(inputs, inputs_raw_length, targets)

                dev_loss_value = loss.item()
                dev_loss += 1.0 * dev_loss_value / dev_dataset.__len__()
                # given ys_hat and targets, evaluate the model
                batch_l_dist, batch_n_token = CER(targets, ys_hat, lens, show_results=show_results, show_num=dev_show_num)
                dev_total_l_dist += batch_l_dist
                dev_total_n_token += batch_n_token

        logging.info("epoch %d: dev loss: %f"%(epoch_i, dev_loss))
        logging.info("dev_total_l_dist: %d, dev_total_n_token: %d, dev CER: %f"%(dev_total_l_dist, dev_total_n_token,
                                        1.0*dev_total_l_dist/dev_total_n_token))
        print("epoch %d: dev loss: %f"%(epoch_i, dev_loss))
        print("dev CER: %f"%(1.0*dev_total_l_dist/dev_total_n_token))

        # test part
        model.eval()
        test_total_l_dist = 0
        test_total_n_token = 0
        with torch.no_grad():
            for i, (data) in enumerate(test_data_loader):
                if i == len(test_sampler):
                    break
                inputs, targets, inputs_raw_length, targets_raw_length, utt_ids = data
                ys_hat, lens, loss = model(inputs, inputs_raw_length, targets)

                # given ys_hat and targets, evaluate the model
                batch_l_dist, batch_n_token = CER(targets, ys_hat, lens)
                test_total_l_dist += batch_l_dist
                test_total_n_token += batch_n_token
        logging.info("test_total_l_dist: %d, test_total_n_token: %d, eval CER: %f"%(test_total_l_dist, test_total_n_token,
                                        1.0*test_total_l_dist/test_total_n_token))
        print("eval CER: %f"%(1.0*test_total_l_dist/test_total_n_token))

def main():
    parser = argparse.ArgumentParser()
    # general configuration
    parser.add_argument('--verbose', '-V', default=0, type=int,
                        help='Verbose option')
    parser.add_argument('--configfile_path', required=True, type=str,
                        help='configfile path')
    parser.add_argument('--expdir', type=str, required=True,
                        help='Experiment directory')
    parser.add_argument('--logfile', type=str, required=True,
                        help='logging file path')

    args = parser.parse_args()

    # logging info
    if args.verbose > 0:
        logging.basicConfig(filename=args.logfile, filemode='a',
            level=logging.INFO, format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s')
    else:
        logging.basicConfig(filename=args.logfile, filemode='a',
            level=logging.WARN, format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s')
        logging.warning('Skip DEBUG/INFO messages')
    # parse cfgfile
    default_cfg = ConfigParser.ConfigParser()
    logging.info("reading configfile: " + args.configfile_path + "...")
    default_cfg.read(args.configfile_path)
    # all configfile infos are now stored into args.cfg
    args.cfg = default_cfg

    # check CUDA_VISIBLE_DEVICES
    ngpu = args.cfg.getint('common', 'ngpu')
    if ngpu > 0:
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cvd is None:
            logging.warn("CUDA_VISIBLE_DEVICES is not set.")
        elif args.ngpu != len(cvd.split(",")):
            logging.error("#gpus is not matched with CUDA_VISIBLE_DEVICES.")
            sys.exit(1)
    # check cuda availability
    if not torch.cuda.is_available():
        logging.warning('cuda is not available')

    # set np random seed
    random_seed = args.cfg.getint('common', 'random_seed')
    if random_seed != -1:
        np.random.seed(random_seed)
        random.seed(random_seed)
    # set pytorch random seed
    torch_random_seed = args.cfg.getint('common', 'torch_random_seed')
    if torch_random_seed != -1:
        torch.manual_seed(torch_random_seed)

    # display cfgfile infos
    for sec in args.cfg.sections():
        for name, value in args.cfg.items(sec):
            logging.info('%s=%s'%(name, value))
    logging.info("=" * 80 + "\n")
    logging.info("start training...")
    asr_train(args)

if __name__ == '__main__':
    main()

    