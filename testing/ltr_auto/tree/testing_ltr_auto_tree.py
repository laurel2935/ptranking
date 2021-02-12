#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Description
A simple script for testing either in-built methods or newly added methods with automatic parameter-tuning
"""

import os
import json
import optuna
import numpy as np

import torch

from ptranking.ltr_global import ltr_seed
from ptranking.ltr_auto.tree.ltr_auto_tree import AutoTreeLTREvaluator

np.random.seed(seed=ltr_seed)


if __name__ == '__main__':

    """
    >>> Tree-based Learning-to-Rank Models <<<
    
    (3) Tree-based Model
    -----------------------------------------------------------------------------------------
    | LightGBMLambdaMART                                                                    |
    -----------------------------------------------------------------------------------------
    
    >>> Supported Datasets <<<
    -----------------------------------------------------------------------------------------
    | LETTOR    | MQ2007_Super %  MQ2008_Super %  MQ2007_Semi %  MQ2008_Semi                |
    -----------------------------------------------------------------------------------------
    | MSLRWEB   | MSLRWEB10K %  MSLRWEB30K                                                  |
    -----------------------------------------------------------------------------------------
    | Yahoo_LTR | Set1 % Set2                                                               |
    -----------------------------------------------------------------------------------------
    | ISTELLA_LTR | Istella_S | Istella | Istella_X                                         |
    -----------------------------------------------------------------------------------------
    
    """

    cuda = None  # the gpu id, e.g., 0 or 1, otherwise, set it as None indicating to use cpu

    debug = True  # in a debug mode, we just check whether the model can operate

    config_with_json = True  # specify configuration with json files or not

    global_study = optuna.create_study(direction='maximize')
    auto_evaluator = AutoTreeLTREvaluator(cuda=cuda)

    if config_with_json:  # specify configuration with json files
        # the directory of json files
        # dir_json = '/Users/dryuhaitao/WorkBench/Dropbox/CodeBench/GitPool/wildltr_ptranking/testing/ltr_adhoc/json/'
        # dir_json = '/Volumes/data_hdd/ptranking.github.io/testing/ltr_adhoc/json/'
        # dir_json = '/Users/solar/WorkBench/Dropbox/CodeBench/GitPool/wildltr_ptranking/testing/ltr_adhoc/json/'
<<<<<<< HEAD
        # dir_json = '/Volumes/data_hdd/ptranking/testing/ltr_auto/tree/json/'
        dir_json = '/Users/iimac/II-Research Dropbox/Hai-Tao Yu/CodeBench/GitPool/auto_ptr/testing/ltr_auto/tree/json/'
=======
        dir_json = '/Volumes/data_hdd/ptranking/testing/ltr_auto/tree/json/'
>>>>>>> update ltr_auto

        auto_evaluator.run(global_study=global_study, auto_evaluator=auto_evaluator, debug=debug,
                               model_id='LightGBMLambdaMART', config_with_json=True, dir_json=dir_json)
    else:
        # data_id = 'MQ2007_Super'
        data_id = 'MQ2008_Super'

        ''' location of the adopted data '''
        # dir_data = '/Users/dryuhaitao/WorkBench/Corpus/' + 'LETOR4.0/MQ2007/'
        # dir_data = '/home/dl-box/WorkBench/Datasets/L2R/LETOR4.0/MQ2007/'
        #dir_data = '/Users/solar/WorkBench/Datasets/L2R/LETOR4.0/MQ2008/'
        dir_data = '/Volumes/data_hdd/dataset/MQ2008/'

        ''' output directory '''
        # dir_output = '/Users/dryuhaitao/WorkBench/CodeBench/Bench_Output/NeuralLTR/Listwise/'
        # dir_output = '/home/dl-box/WorkBench/CodeBench/PyCharmProject/Project_output/Out_L2R/Listwise/'
        # dir_output = '/Users/solar/WorkBench/CodeBench/PyCharmProject/Project_output/Out_L2R/'
        dir_output = '/Volumes/data_hdd/l2r_output/auto/'

        auto_evaluator.run(global_study=global_study, auto_evaluator=auto_evaluator, debug=debug, model_id='LightGBMLambdaMART',
                               config_with_json=False, data_id=data_id, dir_data=dir_data, dir_output=dir_output)