# -*- coding: utf-8 -*-
# @Author: luyizhou4
# @Date:   2019-01-21 18:26:26
# @Function: decoding            
# @Last Modified time: 2019-01-27 20:04:56

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
import kaldi_io

from torch.utils.data import DataLoader

from data_iterator import JsonDataset, SimpleBatchSampler, SequentialSampler, E2EDataLoader
from model import TIMITArch
from metric_utils import CER, plot_peak_property, prefix_search_CER, softmax

def asr_predict(args):
    Plot_figure = False # plot peak_property figures
    #json path
    dev_json_path = args.cfg.get('data', 'dev_json')
    test_json_path = args.cfg.get('data', 'test_json')

    # other config
    batch_size = args.cfg.getint('data', 'batch_size')
    delta_feats_num = args.cfg.getint('data', 'delta_num')
    num_workers = args.cfg.getint('data', 'num_workers')
    normalized = args.cfg.getboolean('data', 'normalized')
    

    # prepare data iterator for dev set
    dev_dataset = JsonDataset(dev_json_path, dataset_type='dev', sorted_type='none', \
                                delta_feats_num=delta_feats_num, normalized=normalized, add_noise=False)
    dev_sampler = SimpleBatchSampler(sampler=SequentialSampler(dev_dataset), 
                                    batch_size=batch_size,
                                    drop_last=False)
    dev_data_loader = E2EDataLoader(dataset=dev_dataset, 
                                    batch_sampler=dev_sampler,
                                    num_workers=num_workers)
    # test data set 
    test_dataset = JsonDataset(test_json_path, dataset_type='test', sorted_type='none', \
                                delta_feats_num=delta_feats_num, normalized=normalized, add_noise=False)
    test_sampler = SimpleBatchSampler(sampler=SequentialSampler(test_dataset), 
                                    batch_size=batch_size,
                                    drop_last=False)
    test_data_loader = E2EDataLoader(dataset=test_dataset, 
                                    batch_sampler=test_sampler,
                                    num_workers=num_workers)

    # get input and output dimension info
    idim = dev_dataset.idim 
    odim = dev_dataset.odim
    hidden_dim = args.cfg.getint('model', 'hidden_dim')
    hidden_layer = args.cfg.getint('model', 'hidden_layer')
    dropout_rate = args.cfg.getfloat('model', 'dropout_rate')

    # construct models
    model = TIMITArch(idim= idim, 
                    hidden_dim=hidden_dim, 
                    hidden_layers=hidden_layer, 
                    odim=odim, 
                    dropout_rate=dropout_rate, 
                    label_ignore_id=-1)

    # init models
    if args.cfg.getint('common', 'ngpu') <= 0:
        model.load_state_dict(torch.load(args.cfg.get('checkpoint', 'checkpoint_path'), map_location='cpu'))
    else:
        model.load_state_dict(torch.load(args.cfg.get('checkpoint', 'checkpoint_path')))
    print("Successfully load params from checkpoint")
    model.eval()
    # set torch device
    device = torch.device("cuda" if args.cfg.getint('common', 'ngpu') > 0 else "cpu")
    model = model.to(device)
    # used for best model selection
    dev_loss = 0.0
    dev_total_l_dist = 0
    dev_total_n_token = 0
    dev_total_per = 0.0
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
            # show peak_prob property
            if i == 5 and Plot_figure:
                plot_peak_property(targets, ys_hat.cpu(), lens.cpu())
            # given ys_hat and targets, evaluate the model
            # print('dev batch %d decoding:'%(i))
            batch_l_dist, batch_n_token = CER(targets, ys_hat.cpu(), lens.cpu(), show_results=show_results, show_num=dev_show_num)
            dev_total_l_dist += batch_l_dist
            dev_total_n_token += batch_n_token

        dev_total_per = 1.0*dev_total_l_dist/dev_total_n_token
        logging.info("dev loss: %f"%(dev_loss))
        logging.info("dev_total_l_dist: %d, dev_total_n_token: %d, dev PER: %f"%(dev_total_l_dist, dev_total_n_token,
                                        dev_total_per))
        print("dev loss: %f"%(dev_loss))
        print("dev PER: %f"%(dev_total_per))

        # test part
        test_total_l_dist = 0
        test_total_n_token = 0
        # w_fd = kaldi_io.open_or_fd('./data/test_preds.ark','wb')
        with torch.no_grad():
            for i, (data) in enumerate(test_data_loader):
                if i == len(test_sampler):
                    break
                inputs, targets, inputs_raw_length, targets_raw_length, utt_ids = data
                ys_hat, lens, loss = model(inputs, inputs_raw_length, targets)
                
                # write ark files
                # preds = ys_hat.cpu()
                # preds = softmax(preds.transpose(0, 1).numpy())
                # preds_lens = lens.cpu().numpy()
                # for utt_i in range(len(utt_ids)):
                #     kaldi_io.write_mat(w_fd, preds[utt_i, :preds_lens[utt_i], :], utt_ids[utt_i])

                # given ys_hat and targets, evaluate the model
                print('test batch %d decoding:'%(i))
                print(utt_ids)
                batch_l_dist, batch_n_token = prefix_search_CER(targets, ys_hat.cpu(), lens.cpu())
                test_total_l_dist += batch_l_dist
                test_total_n_token += batch_n_token
        
        # w_fd.close()
        logging.info("test_total_l_dist: %d, test_total_n_token: %d, eval PER: %f"%(test_total_l_dist, test_total_n_token,
                                        1.0*test_total_l_dist/test_total_n_token))
        print("eval PER: %f"%(1.0*test_total_l_dist/test_total_n_token))

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
    if os.path.exists(args.configfile_path):
        logging.info("reading configfile: " + args.configfile_path + "...")
        default_cfg.read(args.configfile_path)
    else:
        raise Exception("Config file %s not exists"%(args.configfile_path))
    # all configfile infos are now stored into args.cfg
    args.cfg = default_cfg

    # check CUDA_VISIBLE_DEVICES
    ngpu = args.cfg.getint('common', 'ngpu')
    if ngpu > 0:
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cvd is None:
            logging.warn("CUDA_VISIBLE_DEVICES is not set.")
        elif ngpu != len(cvd.split(",")):
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
    logging.info("start evaluation...")
    asr_predict(args)

if __name__ == '__main__':
    main()

    