#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Created by Hai-Tao Yu | 23/08/2020 | https://ii-research.github.io
from ptranking.ltr_tree.lambdamart.lightgbm_lambdaMART import LightGBMLambdaMARTParameter 

class LightGBMLambdaMARTAutoParameter(LightGBMLambdaMARTParameter):
    ''' Parameter class for  WassRank'''
    def __init__(self, debug=False, para_json=None):
        super(LightGBMLambdaMARTAutoParameter, self).__init__(debug=debug, para_json=para_json)

    def default_para_dict(self):
        raise NotImplementedError

    def grid_search(self, trial):
        custom_dict = dict(custom=True, custom_obj_id='lambdarank', use_LGBMRanker=False)
        if self.use_json:
            choice_BT = self.json_dict['BT']
            choice_metric = self.json_dict['metric']
            choice_leaves = self.json_dict['leaves']
            choice_trees = self.json_dict['trees']
            choice_MiData = self.json_dict['MiData']
            choice_MSH = self.json_dict['MSH']
            choice_LR = self.json_dict['LR']
        else:
            # common setting when using in-built lightgbm's ranker
            choice_BT = ['gbdt'] if self.debug else ['gbdt']
            choice_metric = ['ndcg'] if self.debug else ['ndcg']
            choice_leaves = [400] if self.debug else [400]
            choice_trees = [1000] if self.debug else [1000]
            choice_MiData = [50] if self.debug else [50]
            choice_MSH = [200] if self.debug else [200]
            choice_LR = [0.05, 0.01] if self.debug else [0.01, 0.05]

        BT = trial.suggest_categorical('BT', choice_BT)
        metric = trial.suggest_categorical('metric', choice_metric)
        leaves = trial.suggest_int('leaves', choice_leaves[0], choice_leaves[-1], step=100) # low, high, step
        trees = trial.suggest_int('trees', choice_trees[0], choice_trees[-1], step=100) # low, high, step
        MiData = trial.suggest_int('MiData', choice_MiData[0], choice_MiData[-1], step=10) # low, high, step
        MSH = trial.suggest_int('MSH', choice_MSH[0], choice_MSH[-1], step=100) # low, high, step
        LR = trial.suggest_float('LR', choice_LR[0], choice_LR[-1], step=0.01) # low, high, step


        lightgbm_para_dict = {'boosting_type': BT,  # ltr_gbdt, dart
                                     'objective': 'lambdarank',
                                     'metric': metric,
                                     'learning_rate': LR,
                                     'num_leaves': leaves,
                                     'num_trees': trees,
                                     'num_threads': 16,
                                     'min_data_in_leaf': MiData,
                                     'min_sum_hessian_in_leaf': MSH,
                                     # 'lambdamart_norm':False,
                                     # 'is_training_metric':True,
                                     'verbosity': -1}


        self.para_dict = dict(custom_dict=custom_dict, lightgbm_para_dict=lightgbm_para_dict)

        return self.para_dict
