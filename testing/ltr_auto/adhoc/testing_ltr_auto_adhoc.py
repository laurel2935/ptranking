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
from ptranking.ltr_auto.adhoc.ltr_auto_adhoc import AutoLTREvaluator

np.random.seed(seed=ltr_seed)
torch.manual_seed(seed=ltr_seed)

if __name__ == '__main__':
    """
    >>> Learning-to-Rank Models <<<
    (1) Optimization based on Empirical Risk Minimization
    -----------------------------------------------------------------------------------------
    | Pointwise | RankMSE                                                                   |
    -----------------------------------------------------------------------------------------
    | Pairwise  | RankNet                                                                   |
    -----------------------------------------------------------------------------------------
    | Listwise  | LambdaRank % ListNet % ListMLE % RankCosine %  ApproxNDCG %  WassRank     |
    |           | STListNet                                                                 |
    -----------------------------------------------------------------------------------------   


    >>> Supported Datasets <<<
    -----------------------------------------------------------------------------------------
    | LETTOR    | MQ2007_Super %  MQ2008_Super %  MQ2007_Semi %  MQ2008_Semi                |
    -----------------------------------------------------------------------------------------
    | MSLRWEB   | MSLRWEB10K %  MSLRWEB30K                                                  |
    -----------------------------------------------------------------------------------------
    | Yahoo_LTR | Set1 % Set2                                                               |
    -----------------------------------------------------------------------------------------
    | ISTELLA_LTR | Istella_S % Istella % Istella_X                                         |
    -----------------------------------------------------------------------------------------

    """

    cuda = None  # the gpu id, e.g., 0 or 1, otherwise, set it as None indicating to use cpu

    debug = False  # in a debug mode, we just check whether the model can operate

    config_with_json = True  # specify configuration with json files or not

    models_to_run = [
        # 'RankMSE',
        # 'RankNet',
        'LambdaRank',
        # 'ListNet',
        # 'ListMLE',
        # 'RankCosine',
        # 'ApproxNDCG',
        # 'WassRank',
        # 'STListNet',
        # 'LambdaLoss'
    ]

    global_study = optuna.create_study(direction='maximize')
    auto_evaluator = AutoLTREvaluator(cuda=cuda)

    if config_with_json:  # specify configuration with json files
        # the directory of json files
        # dir_json = '/Users/dryuhaitao/WorkBench/Dropbox/CodeBench/GitPool/wildltr_ptranking/testing/ltr_adhoc/json/'
        # dir_json = '/Volumes/data_hdd/ptranking.github.io/testing/ltr_adhoc/json/'
        # dir_json = '/Users/solar/WorkBench/Dropbox/CodeBench/GitPool/wildltr_ptranking/testing/ltr_adhoc/json/'
        dir_json = '/Volumes/data_hdd/ptranking/testing/ltr_auto/adhoc/json/'

        for model_id in models_to_run:
            auto_evaluator.run(global_study=global_study, auto_evaluator=auto_evaluator, debug=debug,
                               model_id=model_id, config_with_json=True, dir_json=dir_json)
    else:
        # data_id = 'MQ2007_Super'
        data_id = 'MQ2008_Super'

        ''' location of the adopted data '''
        # dir_data = '/Users/dryuhaitao/WorkBench/Corpus/' + 'LETOR4.0/MQ2007/'
        # dir_data = '/home/dl-box/WorkBench/Datasets/L2R/LETOR4.0/MQ2007/'
        #dir_data = '/Users/solar/WorkBench/Datasets/L2R/LETOR4.0/MQ2008/'
        dir_data = '/Users/iimac/Workbench/Corpus/L2R/LETOR4.0/MQ2008/'

        ''' output directory '''
        # dir_output = '/Users/dryuhaitao/WorkBench/CodeBench/Bench_Output/NeuralLTR/Listwise/'
        # dir_output = '/home/dl-box/WorkBench/CodeBench/PyCharmProject/Project_output/Out_L2R/Listwise/'
        # dir_output = '/Users/solar/WorkBench/CodeBench/PyCharmProject/Project_output/Out_L2R/'
        dir_output = '/Users/iimac/Workbench/CodeBench/Output/NeuralLTR/'

        for model_id in models_to_run:
            auto_evaluator.run(global_study=global_study, auto_evaluator=auto_evaluator, debug=debug, model_id=model_id,
                               config_with_json=False, data_id=data_id, dir_data=dir_data, dir_output=dir_output)