# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import argparse
import os
import re
import sys
import subprocess

if sys.version_info >= (3, 0):
    import configparser
else:
    import ConfigParser as configparser

def parse_args(file_path):
    default_cfg = configparser.ConfigParser()
    print("[    INFO] " + "reading configfile: " + file_path + "...")
    default_cfg.read(file_path)

    parser = argparse.ArgumentParser()
    parser.add_argument("--configfile", help="config file")
    parser.add_argument("--archfile", help="symbol architecture template file")
    # those allow us to overwrite the configs through command line
    # egs: --common_mode
    for sec in default_cfg.sections():
        for name, _ in default_cfg.items(sec):
            arg_name = '--%s_%s' % (sec, name)
            doc = 'Overwrite %s in section [%s] of config file' % (name, sec)
            parser.add_argument(arg_name, help=doc)

    args = parser.parse_args()
    if args.configfile is not None:
        # now read the user supplied config file to overwrite some values
        default_cfg.read(args.configfile)
    else:
        raise Exception('cfg file path must be provided. ' +
                        'ex)python main.py --configfile examplecfg.cfg' +
                        '(warning)--configfile examplecfg.cfg must be the first argument')

    # now overwrite config from command line options
    for sec in default_cfg.sections():
        for name, _ in default_cfg.items(sec):
            arg_name = ('%s_%s' % (sec, name))
            if hasattr(args, arg_name) and getattr(args, arg_name) is not None:
                if str(getattr(args, arg_name)) == default_cfg.get(sec, name):
                    sys.stderr.write('Warning: CMDLine and config are the same: %s\n' % (str(getattr(args, arg_name))))
                else:
                    sys.stderr.write('Warning: CMDLine overwriting %s.%s:\n' % (sec, name))
                    sys.stderr.write("    '%s' => '%s'\n" % (default_cfg.get(sec, name),
                                                             getattr(args, arg_name)))
                    default_cfg.set(sec, name, getattr(args, arg_name))

    # all new changes in the cmd line are now stored into args.config
    args.config = default_cfg
    sys.stderr.write("=" * 80 + "\n")

    # set archfile to read template of network
    if args.archfile is not None:
        # now read the user supplied config file to overwrite some values
        args.config.set('arch', 'arch_file', args.archfile)
    else:
        # deep speech architecture
        args.config.set('arch', 'arch_file', 'arch_deepspeech')
    return args

def generate_file_path(save_dir, model_name, postfix):
    if os.path.isabs(model_name):
        return os.path.join(model_name, postfix)
    # if it is not a full path it stores under (its project path)/save_dir/model_name/postfix
    return os.path.abspath(os.path.join(os.path.dirname(__file__), save_dir, model_name + '_' + postfix))

def get_checkpoint_path(args):
    prefix = args.config.get('common', 'prefix')
    checkpoint_dir = args.config.get('common', 'checkpoint_dir')
    if os.path.isabs(prefix):
        return prefix
    return os.path.abspath(os.path.join(os.path.dirname('__file__'), checkpoint_dir, prefix))

