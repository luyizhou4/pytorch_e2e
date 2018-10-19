#coding:utf-8

import os
import sys
import logging
import ConfigParser
import argparse

def asr_train(args):
    


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

    # set random seed
    random_seed = args.cfg.getint('common', 'random_seed')
    if random_seed != -1:
        np.random.seed(random_seed)

    # display cfgfile infos
    for sec in args.cfg.sections():
        for name, value in args.cfg.items(sec):
            logging.info('%s=%s'%(name, value))
    logging.info("=" * 80 + "\n")

    asr_train(args)

if __name__ == '__main__':
    main()

    