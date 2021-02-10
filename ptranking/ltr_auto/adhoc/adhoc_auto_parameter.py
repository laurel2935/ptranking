#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Description

"""
import numpy as np

from ptranking.ltr_adhoc.eval.parameter import ModelParameter
from ptranking.ltr_adhoc.pairwise.ranknet import RankNetParameter
from ptranking.ltr_adhoc.listwise.listmle import ListMLEParameter
from ptranking.ltr_adhoc.listwise.lambdarank import LambdaRankParameter
from ptranking.ltr_adhoc.listwise.approxNDCG import ApproxNDCGParameter
from ptranking.ltr_adhoc.listwise.st_listnet import STListNetParameter
from ptranking.ltr_adhoc.listwise.lambdaloss import LambdaLossParameter
from ptranking.ltr_adhoc.listwise.wassrank.wassRank import WassRankParameter

##########
# A general one for parameter-free models
##########

class AutoModelParameter(ModelParameter):
    ''' Parameter class for  parameter-free models '''
    def __init__(self, model_id=None):
        super(AutoModelParameter, self).__init__(model_id=model_id)

    def default_para_dict(self):
        raise NotImplementedError

    def grid_search(self, trial):
        return dict(model_id=self.model_id)

##########
# Pairwise
##########

class RankNetAutoParameter(RankNetParameter):
    ''' Parameter class for  RankNet'''
    def __init__(self, debug=False, para_json=None):
        super(RankNetAutoParameter, self).__init__(debug=debug, para_json=para_json)

    def default_para_dict(self):
        raise NotImplementedError

    def grid_search(self, trial):
        if self.use_json:
            choice_sigma = self.json_dict['sigma']
        else:
            choice_sigma = [1.0] if self.debug else [1.0]

        sigma = trial.suggest_float('sigma', choice_sigma[0], choice_sigma[-1], step=1.0) # low, high, step

        self.ranknet_para_dict = dict(model_id=self.model_id, sigma=sigma)
        return self.ranknet_para_dict

##########
# Listwise
##########

class ListMLEAutoParameter(ListMLEParameter):
    ''' Parameter class for  ListMLE'''
    def __init__(self, debug=False, para_json=None):
        super(ListMLEAutoParameter, self).__init__(debug=debug, para_json=para_json)

    def default_para_dict(self):
        raise NotImplementedError

    def grid_search(self, trial):
        if self.use_json:
            choice_samples_per_query = self.json_dict['samples_per_query']
        else:
            choice_samples_per_query = [1] if self.debug else [1, 5]

        samples_per_query = trial.suggest_int('samples_per_query', choice_samples_per_query[0], choice_samples_per_query[-1], step=1) # low, high, step

        self.listmle_para_dict = dict(model_id=self.model_id, samples_per_query=samples_per_query)
        return self.listmle_para_dict

class LambdaRankAutoParameter(LambdaRankParameter):
    ''' Parameter class for LambdaRank '''
    def __init__(self, debug=False, para_json=None):
        super(LambdaRankAutoParameter, self).__init__(debug=debug, para_json=para_json)

    def default_para_dict(self):
        raise NotImplementedError

    def grid_search(self, trial):
        if self.use_json:
            choice_sigma = self.json_dict['sigma']
        else:
            choice_sigma = [1.0, 5.0] if self.debug else [1.0, 5.0]  # 1.0, 10.0, 50.0, 100.0

        sigma = trial.suggest_float('sigma', choice_sigma[0], choice_sigma[-1], step=1.0) # low, high, step

        self.lambda_para_dict = dict(model_id=self.model_id, sigma=sigma, loss_version='Full')
        return self.lambda_para_dict

class ApproxNDCGAutoParameter(ApproxNDCGParameter):
    ''' Parameter class for  ApproxNDCG'''
    def __init__(self, debug=False, para_json=None):
        super(ApproxNDCGAutoParameter, self).__init__(debug=debug, para_json=para_json)

    def default_para_dict(self):
        raise NotImplementedError

    def grid_search(self, trial):
        if self.use_json:
            choice_alpha = self.json_dict['alpha']
        else:
            choice_alpha = [10.0] if self.debug else [10.0]  # 1.0, 10.0, 50.0, 100.0

        alpha = trial.suggest_float('alpha', choice_alpha[0], choice_alpha[-1], step=1.0) # low, high, step

        self.apxNDCG_para_dict = dict(model_id=self.model_id, alpha=alpha)
        return self.apxNDCG_para_dict


class STListNetAutoParameter(STListNetParameter):
    ''' Parameter class for  STListNet'''
    def __init__(self, debug=False, para_json=None):
        super(STListNetAutoParameter, self).__init__(debug=debug, para_json=para_json)

    def default_para_dict(self):
        raise NotImplementedError

    def grid_search(self, trial):
        if self.use_json:
            choice_temperature = self.json_dict['temperature']
        else:
            choice_temperature = [1.0] if self.debug else [1.0]

        temperature = trial.suggest_float('temperature', choice_temperature[0], choice_temperature[-1], step=1.0) # low, high, step

        self.stlistnet_para_dict = dict(model_id=self.model_id, temperature=temperature)
        return self.stlistnet_para_dict

class LambdaLossAutoParameter(LambdaLossParameter):
    ''' Parameter class for LambdaLoss '''
    def __init__(self, debug=False, para_json=None):
        super(LambdaLossAutoParameter, self).__init__(debug=debug, para_json=para_json)

    def default_para_dict(self):
        raise NotImplementedError

    def grid_search(self, trial):
        if self.use_json:
            choice_loss_type = self.json_dict['loss_type']
            choice_sigma = self.json_dict['sigma']
            choice_mu = self.json_dict['mu']
            choice_k = self.json_dict['k']
        else: 
            choice_loss_type = ['NDCG_Loss2'] if self.debug else ['NDCG_Loss2']
            choice_sigma = [1.0] if self.debug else [1.0]  #
            choice_mu = [5.0] if self.debug else [5.0]  #
            choice_k = [5.0] if self.debug else [5.0]

        loss_type = trial.suggest_categorical('loss_type', choice_loss_type)
        sigma = trial.suggest_float('sigma', choice_sigma[0], choice_sigma[-1], step=1.0) # low, high, step
        mu = trial.suggest_float('mu', choice_mu[0], choice_mu[-1], step=1.0) # low, high, step
        k = trial.suggest_float('k', choice_k[0], choice_k[-1], step=1.0) # low, high, step

        if 'NDCG_Loss2++' == loss_type:
            self.lambdaloss_para_dict = dict(model_id=self.model_id, sigma=sigma, loss_type = loss_type, mu=mu, k=k)

        else:
            self.lambdaloss_para_dict = dict(model_id=self.model_id, sigma=sigma, loss_type=loss_type, k=k)

        return self.lambdaloss_para_dict

class WassRankAutoParameter(WassRankParameter):
    ''' Parameter class for  WassRank'''
    def __init__(self, debug=False, para_json=None):
        super(WassRankAutoParameter, self).__init__(debug=debug, para_json=para_json)

    def default_para_dict(self):
        raise NotImplementedError

    def grid_search(self, trial):
        if self.use_json:
            wass_choice_mode = self.json_dict['mode']
            wass_choice_itr = self.json_dict['itr']
            wass_choice_lam = self.json_dict['lam']

            wass_cost_type = self.json_dict['cost_type']
            # member parameters of 'Group' include margin, div, group-base
            wass_choice_non_rele_gap = self.json_dict['non_rele_gap']
            wass_choice_var_penalty = self.json_dict['var_penalty']
            wass_choice_group_base = self.json_dict['group_base']

            wass_choice_smooth = self.json_dict['smooth']
            wass_choice_norm = self.json_dict['norm']
        else:
            wass_choice_mode = ['WassLossSta']  # EOTLossSta | WassLossSta
            wass_choice_itr = [10]  # number of iterations w.r.t. sink-horn operation
            wass_choice_lam = [0.1]  # 0.01 | 1e-3 | 1e-1 | 10  regularization parameter

            wass_cost_type = ['eg']  # p1 | p2 | eg | dg| ddg
            # member parameters of 'Group' include margin, div, group-base
            wass_choice_non_rele_gap = [10]  # the gap between a relevant document and an irrelevant document
            wass_choice_var_penalty = [np.e]  # variance penalty
            wass_choice_group_base = [4]  # the base for computing gain value

            wass_choice_smooth = ['ST']  # 'ST', i.e., ST: softmax | Gain, namely the way on how to get the normalized distribution histograms
            wass_choice_norm = ['BothST']  # 'BothST': use ST for both prediction and standard labels

        mode = trial.suggest_categorical('mode', wass_choice_mode)
        itr = trial.suggest_int('itr', wass_choice_itr[0], wass_choice_itr[-1], step=1) # low, high, step
        lam = trial.suggest_float('lam', wass_choice_lam[0], wass_choice_lam[-1], step=1.0) # low, high, step
        cost_type = trial.suggest_categorical('cost_type', wass_cost_type)
        non_rele_gap = trial.suggest_int('non_rele_gap', wass_choice_non_rele_gap[0], wass_choice_non_rele_gap[-1], step=1) # low, high, step
        var_penalty = trial.suggest_float('var_penalty', wass_choice_var_penalty[0], wass_choice_var_penalty[-1], step=1) # low, high, step
        group_base = trial.suggest_int('group_base', wass_choice_group_base[0], wass_choice_group_base[-1], step=1) # low, high, step
        smooth = trial.suggest_categorical('smooth', wass_choice_smooth)
        norm = trial.suggest_categorical('norm', wass_choice_norm)


        self.wass_para_dict = dict(model_id='WassRank', mode=mode, sh_itr=itr, lam=lam,
                                                   cost_type=cost_type, smooth_type=smooth, norm_type=norm,
                                                   gain_base=group_base, non_rele_gap=non_rele_gap, var_penalty=var_penalty)
        return self.wass_para_dict
