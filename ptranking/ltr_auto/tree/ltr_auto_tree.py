import torch
import os
import sys
import optuna
import json
from ptranking.ltr_adhoc.eval.eval_utils import ndcg_at_k
from ptranking.ltr_adhoc.eval.ltr import LTREvaluator
from ptranking.ltr_tree.eval.ltr_tree import TreeLTREvaluator
from ptranking.ltr_auto.base.auto_tree_parameter import AutoTreeDataSetting, AutoTreeEvalSetting
from ptranking.ltr_auto.tree.tree_auto_parameter import LightGBMLambdaMARTAutoParameter 
import json
import numpy as np
from itertools import product
from sklearn.datasets import load_svmlight_file

import lightgbm as lgbm
from lightgbm import Dataset

from ptranking.data.data_utils import load_letor_data_as_libsvm_data, YAHOO_LTR, SPLIT_TYPE, ISTELLA_LTR
from ptranking.ltr_tree.util.lightgbm_util import \
    lightgbm_custom_obj_lambdarank, lightgbm_custom_obj_ranknet, lightgbm_custom_obj_listnet,\
    lightgbm_custom_obj_lambdarank_fobj, lightgbm_custom_obj_ranknet_fobj, lightgbm_custom_obj_listnet_fobj
from ptranking.metric.adhoc_metric import torch_nDCG_at_ks, torch_nerr_at_ks, torch_ap_at_ks, torch_precision_at_ks

class TreeLTRObjective(object):
    """
    The customized optuna objective over training data and validation data
    """

    def __init__(self, model_id=None, data_id=None, x_train=None , y_train=None, group_train=None, train_set=None, x_valid=None , y_valid=None, group_valid=None, valid_set=None, ranker=None, fold_k =None, para_dict=None, data_dict=None, eval_dict=None, save_model_dir =None):
        self.model_id = model_id
        self.custom_dict = para_dict['custom_dict']
        self.lightgbm_para_dict = para_dict['lightgbm_para_dict']
        self.data_id = data_id
        self.x_train = x_train
        self.y_train = y_train
        self.group_train = group_train
        self.train_set = train_set
        self.x_valid = x_valid
        self.y_valid = y_valid
        self.group_valid = group_valid
        self.valid_set = valid_set
        self.ranker = ranker
        self.fold_k = fold_k
        self.data_dict = data_dict
        self.eval_dict = eval_dict
        self.save_model_dir = save_model_dir
    
    def get_custom_obj(self, custom_obj_id, fobj=False):
        if fobj:
            if custom_obj_id == 'ranknet':
                return lightgbm_custom_obj_ranknet_fobj
            elif custom_obj_id == 'listnet':
                return lightgbm_custom_obj_listnet_fobj
            elif custom_obj_id == 'lambdarank':
                return lightgbm_custom_obj_lambdarank_fobj
            else:
                raise NotImplementedError
        else:
            if custom_obj_id == 'ranknet':
                return lightgbm_custom_obj_ranknet
            elif custom_obj_id == 'listnet':
                return lightgbm_custom_obj_listnet
            elif custom_obj_id == 'lambdarank':
                return lightgbm_custom_obj_lambdarank
            else:
                raise NotImplementedError
    
    def cal_metric_at_ks(self, model_id, all_std_labels=None, all_preds=None, group=None, ks=[1, 3, 5, 10], label_type=None):
        """
        Compute metric values with different cutoff values
        :param model:
        :param all_std_labels:
        :param all_preds:
        :param group:
        :param ks:
        :return:
        """
        cnt = torch.zeros(1)

        sum_ndcg_at_ks = torch.zeros(len(ks))
        sum_nerr_at_ks = torch.zeros(len(ks))
        sum_ap_at_ks = torch.zeros(len(ks))
        sum_p_at_ks = torch.zeros(len(ks))

        list_ndcg_at_ks_per_q = []
        list_err_at_ks_per_q = []
        list_ap_at_ks_per_q = []
        list_p_at_ks_per_q = []

        tor_all_std_labels, tor_all_preds = \
            torch.from_numpy(all_std_labels.astype(np.float32)), torch.from_numpy(all_preds.astype(np.float32))

        head = 0
        if model_id.startswith('LightGBM'): group = group.astype(np.int).tolist()
        for gr in group:
            tor_per_query_std_labels = tor_all_std_labels[head:head+gr]
            tor_per_query_preds = tor_all_preds[head:head+gr]
            head += gr

            _, tor_sorted_inds = torch.sort(tor_per_query_preds, descending=True)

            sys_sorted_labels = tor_per_query_std_labels[tor_sorted_inds]
            ideal_sorted_labels, _ = torch.sort(tor_per_query_std_labels, descending=True)

            ndcg_at_ks = torch_nDCG_at_ks(batch_sys_sorted_labels=sys_sorted_labels.view(1, -1),
                                          batch_ideal_sorted_labels=ideal_sorted_labels.view(1, -1), ks=ks, label_type=label_type)
            ndcg_at_ks = torch.squeeze(ndcg_at_ks, dim=0)
            list_ndcg_at_ks_per_q.append(ndcg_at_ks.numpy())

            nerr_at_ks = torch_nerr_at_ks(batch_sys_sorted_labels=sys_sorted_labels.view(1, -1),
                                          batch_ideal_sorted_labels=ideal_sorted_labels.view(1, -1), ks=ks, label_type=label_type)
            nerr_at_ks = torch.squeeze(nerr_at_ks, dim=0)
            list_err_at_ks_per_q.append(nerr_at_ks.numpy())

            ap_at_ks = torch_ap_at_ks(batch_sys_sorted_labels=sys_sorted_labels.view(1, -1),
                                      batch_ideal_sorted_labels=ideal_sorted_labels.view(1, -1), ks=ks)
            ap_at_ks = torch.squeeze(ap_at_ks, dim=0)
            list_ap_at_ks_per_q.append(ap_at_ks.numpy())

            p_at_ks = torch_precision_at_ks(batch_sys_sorted_labels=sys_sorted_labels.view(1, -1), ks=ks)
            p_at_ks = torch.squeeze(p_at_ks, dim=0)
            list_p_at_ks_per_q.append(p_at_ks.numpy())

            sum_ndcg_at_ks = torch.add(sum_ndcg_at_ks, ndcg_at_ks)
            sum_nerr_at_ks = torch.add(sum_nerr_at_ks, nerr_at_ks)
            sum_ap_at_ks   = torch.add(sum_ap_at_ks, ap_at_ks)
            sum_p_at_ks    = torch.add(sum_p_at_ks, p_at_ks)
            cnt += 1

        tor_avg_ndcg_at_ks = sum_ndcg_at_ks / cnt
        avg_ndcg_at_ks = tor_avg_ndcg_at_ks.data.numpy()

        tor_avg_nerr_at_ks = sum_nerr_at_ks / cnt
        avg_nerr_at_ks = tor_avg_nerr_at_ks.data.numpy()

        tor_avg_ap_at_ks = sum_ap_at_ks / cnt
        avg_ap_at_ks = tor_avg_ap_at_ks.data.numpy()

        tor_avg_p_at_ks = sum_p_at_ks / cnt
        avg_p_at_ks = tor_avg_p_at_ks.data.numpy()

        return avg_ndcg_at_ks, avg_nerr_at_ks, avg_ap_at_ks, avg_p_at_ks,\
               list_ndcg_at_ks_per_q, list_err_at_ks_per_q, list_ap_at_ks_per_q, list_p_at_ks_per_q


    def __call__(self, trial):

        if self.custom_dict['custom'] and self.custom_dict['use_LGBMRanker']:
            lgbm_ranker = self.ranker
            lgbm_ranker.set_params(**self.lightgbm_para_dict)
            '''
            objective : string, callable or None, optional (default=None)
            Specify the learning task and the corresponding learning objective or
            a custom objective function to be used (see note below).
            Default: 'regression' for LGBMRegressor, 'binary' or 'multiclass' for LGBMClassifier, 'lambdarank' for LGBMRanker.
            '''
            custom_obj_dict = dict(objective=self.get_custom_obj(custom_obj_id=self.custom_dict['custom_obj_id']))
            lgbm_ranker.set_params(**custom_obj_dict)
            '''
            eval_set (list or None, optional (default=None)) – A list of (X, y) tuple pairs to use as validation sets.
            cf. https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRanker.html
            '''
            lgbm_ranker.fit(self.x_train, self.y_train, group=self.group_train,
                            eval_set=[(self.x_valid, self.y_valid)], eval_group=[self.group_valid], eval_at=[5],
                            early_stopping_rounds=self.eval_dict['epochs'],
                            verbose=10)

        elif self.custom_dict['custom']:
            # use the argument of fobj
            lgbm_ranker = self.ranker.train(params=self.lightgbm_para_dict, verbose_eval=10,
                                        train_set=self.train_set, valid_sets=[self.valid_set],
                                        early_stopping_rounds=self.eval_dict['epochs'],
                                        fobj=self.get_custom_obj(custom_obj_id=self.custom_dict['custom_obj_id'],
                                                                fobj=True))
        else: # trained booster as ranker
            lgbm_ranker = self.ranker.train(params=self.lightgbm_para_dict, verbose_eval=10,
                                        train_set=self.train_set, valid_sets=[self.valid_set],
                                        early_stopping_rounds=self.eval_dict['epochs'])

        
        if self.data_id in YAHOO_LTR:
            model_file = self.save_model_dir + 'model.txt'
        else:
            model_file = self.save_model_dir + '_'.join(['fold', str(self.fold_k), 'model'])+'.txt'

        if self.custom_dict['custom'] and self.custom_dict['use_LGBMRanker']:
            lgbm_ranker.booster_.save_model(model_file)
        else:
            lgbm_ranker.save_model(model_file)
        
        y_pred = lgbm_ranker.predict(self.x_valid)

        fold_avg_ndcg_at_ks, fold_avg_nerr_at_ks, fold_avg_ap_at_ks, fold_avg_p_at_ks,list_ndcg_at_ks_per_q, list_err_at_ks_per_q, list_ap_at_ks_per_q, list_p_at_ks_per_q = \
            self.cal_metric_at_ks(model_id=self.model_id, all_std_labels=self.y_valid, all_preds=y_pred, group=self.group_valid, ks=[1, 3, 5, 10], label_type=self.data_dict['label_type'])
        

        vali_eval_v = fold_avg_ndcg_at_ks[2]

        return vali_eval_v


class AutoTreeLTREvaluator(TreeLTREvaluator):
    """
    The class for evaluating different adversarial adversarial ltr methods.
    """
    def __init__(self, frame_id='AutoLTR', cuda=None):
        super(AutoTreeLTREvaluator, self).__init__(frame_id=frame_id, cuda=cuda)


    def setup_output(self, data_dict=None, eval_dict=None):
        """
        Update output directory
        :param data_dict:
        :param eval_dict:
        :param sf_para_dict:
        :param model_para_dict:
        :return:
        """
        model_id = self.model_parameter.model_id
        grid_search, do_vali, dir_output = eval_dict['grid_search'], eval_dict['do_validation'], eval_dict['dir_output']
        mask_label = eval_dict['mask_label']

        if grid_search:
            dir_root = dir_output + '_'.join(['gpu', 'grid', model_id]) + '/' if self.gpu else dir_output + '_'.join(['grid', model_id]) + '/'
        else:
            dir_root = dir_output

        eval_dict['dir_root'] = dir_root

        if not os.path.exists(dir_root): os.makedirs(dir_root)

        data_eval_str = '_'.join([self.data_setting.to_data_setting_string(),
                                  self.eval_setting.to_eval_setting_string()])
        if mask_label:
            data_eval_str = '_'.join([data_eval_str, 'MaskLabel', 'Ratio', '{:,g}'.format(eval_dict['mask_ratio'])])

        file_prefix = '_'.join([model_id, data_eval_str])

        if data_dict['scale_data']:
            if data_dict['scaler_level'] == 'QUERY':
                file_prefix = '_'.join([file_prefix, 'QS', data_dict['scaler_id']])
            else:
                file_prefix = '_'.join([file_prefix, 'DS', data_dict['scaler_id']])

        dir_run = dir_root + file_prefix + '/'  # run-specific outputs

        eval_dict['dir_run'] = dir_run
        if not os.path.exists(dir_run):
            os.makedirs(dir_run)

        return dir_run

    def setup_auto_kfold_cv_eval(self, debug=False, model_id=None, dir_json=None, data_id=None, dir_data=None, dir_output=None):
        if dir_json is not None:
            # eval_json = dir_json + 'EvalSetting.json'
            # data_json = dir_json + 'DataSetting.json'
            # sf_json   = dir_json + 'SFParameter.json'
            ad_data_eval_sf_json = dir_json + 'Auto_Tree_Data_Eval_ScoringFunction.json'
            para_json = dir_json + model_id + "AutoParameter.json"

            self.set_eval_setting(debug=debug, tree_eval_json=ad_data_eval_sf_json)
            self.set_data_setting(tree_data_json=ad_data_eval_sf_json)
            self.set_model_setting(model_id=model_id, para_json=para_json)
        else:
            self.set_eval_setting(debug=debug, dir_output=dir_output)
            self.set_data_setting(debug=debug, data_id=data_id, dir_data=dir_data)
            self.set_model_setting(debug=debug, model_id=model_id)

        self.data_dict = self.get_default_data_setting()
        self.eval_dict = self.get_default_eval_setting()

        self.declare_global(model_id=model_id)

        # update data_meta given the debug mode
        if self.eval_dict['debug']: self.data_dict['fold_num'], self.eval_dict['epochs'] = 1, 2  # for quick check
        if self.data_dict['data_id'] == 'IRGAN_MQ2008_Semi': self.data_dict['fold_num'] = 1

        self.dir_run = self.setup_output(self.data_dict, self.eval_dict)

        # for quick access of common evaluation settings
        self.fold_num = self.data_dict['fold_num']
        self.epochs = self.eval_dict['epochs']
        self.vali_k, self.cutoffs = self.eval_dict['vali_k'], self.eval_dict['cutoffs']

        #if self.do_log: sys.stdout = open(self.dir_run + 'log.txt', "w")

        self.k_studies = [optuna.create_study(direction='maximize') for _ in range(self.fold_num)]

    def set_data_setting(self, tree_data_json=None, debug=False, data_id=None, dir_data=None):
        if tree_data_json is not None:
            self.data_setting = AutoTreeDataSetting(tree_data_json=tree_data_json)
        else:
            self.data_setting = AutoTreeDataSetting(debug=debug, data_id=data_id, dir_data=dir_data)

    def set_eval_setting(self, tree_eval_json=None, debug=False, dir_output=None):
        if tree_eval_json is not None:
            self.eval_setting = AutoTreeEvalSetting(debug=debug, tree_eval_json=tree_eval_json)
        else:
            self.eval_setting = AutoTreeEvalSetting(debug=debug, dir_output=dir_output)


    def set_model_setting(self, model_id=None, para_json=None, debug=False):
        if para_json is not None:
            self.model_parameter = globals()[model_id + "AutoParameter"](para_json=para_json)
        else: # the 3rd type, where debug-mode enables quick test
            self.model_parameter = globals()[model_id + "AutoParameter"](debug=debug)


    def __call__(self, trial):
        # sample params
        model_id = self.model_parameter.model_id
        para_dict = self.model_parameter.grid_search(trial)
        self.setup_eval(data_dict=self.data_dict, eval_dict=self.eval_dict)

        k_flod_average = 0.
        for i in range(self.fold_num):  # evaluation over k-fold data
            fold_k = i+1
            study = self.k_studies[i]

            train_data, test_data, vali_data = self.load_data(self.eval_dict, self.data_dict, fold_k)

            data_id = self.data_dict['data_id']

            train_presort, validation_presort, test_presort = self.data_dict['train_presort'], self.data_dict['validation_presort'],\
                                                                self.data_dict['test_presort']

            file_train, file_vali, file_test = self.determine_files(data_dict=self.data_dict, fold_k=fold_k)

            self.update_save_model_dir(data_dict=self.data_dict, fold_k=fold_k)

            # prepare training & testing datasets
            file_train_data, file_train_group = load_letor_data_as_libsvm_data(file_train, split_type=SPLIT_TYPE.Train,
                                                            data_dict=self.data_dict, eval_dict=self.eval_dict, presort=train_presort)
            x_train, y_train = load_svmlight_file(file_train_data)
            group_train = np.loadtxt(file_train_group)
            train_set = Dataset(data=x_train, label=y_train, group=group_train)

            file_test_data, file_test_group = load_letor_data_as_libsvm_data(file_test, split_type=SPLIT_TYPE.Test,
                                                            data_dict=self.data_dict, eval_dict=self.eval_dict, presort=test_presort)
            x_test, y_test = load_svmlight_file(file_test_data)
            group_test = np.loadtxt(file_test_group)

            file_vali_data, file_vali_group=load_letor_data_as_libsvm_data(file_vali, split_type=SPLIT_TYPE.Validation,
                                                    data_dict=self.data_dict, eval_dict=self.eval_dict, presort=validation_presort)
            x_valid, y_valid = load_svmlight_file(file_vali_data)
            group_valid = np.loadtxt(file_vali_group)
            valid_set = Dataset(data=x_valid, label=y_valid, group=group_valid)

            if para_dict['custom_dict']['custom'] and para_dict['custom_dict']['use_LGBMRanker']:
                lgbm_ranker = lgbm.LGBMRanker()
            else:
                lgbm_ranker = lgbm

            study.optimize(TreeLTRObjective(model_id=model_id, data_id = data_id, x_train=x_train , y_train=y_train, group_train=group_train, train_set=train_set, x_valid=x_valid , y_valid=y_valid, group_valid=group_valid, valid_set=valid_set, \
                ranker=lgbm_ranker, fold_k=fold_k, para_dict=para_dict, data_dict=self.data_dict, eval_dict=self.eval_dict, save_model_dir=self.save_model_dir),
                           n_trials=1) # ??? the meaning of n_trials
            # store loss
            if data_id in YAHOO_LTR:
                model_file = self.save_model_dir + 'model.txt'
            else:
                model_file = self.save_model_dir + '_'.join(['fold', str(fold_k), 'model'])+'.txt'
            
            lgbm_ranker = lgbm.Booster(model_file=model_file)

            vali_eval_tmp = ndcg_at_k(ranker=lgbm_ranker, test_data=vali_data, k=self.vali_k,
                                      label_type=vali_data.label_type, gpu=self.gpu, device=self.device)
            vali_eval_v = vali_eval_tmp.data.numpy()
            k_flod_average += vali_eval_v

        # calculate loss todo average k-fold validation score
        k_flod_average /= self.fold_num

        return k_flod_average
    
    def run(self, global_study=None, auto_evaluator=None, debug=False,
            model_id=None, config_with_json=None, dir_json=None, data_id=None, dir_data=None, dir_output=None):
        if config_with_json:
            assert dir_json is not None
            auto_evaluator.setup_auto_kfold_cv_eval(debug=debug, model_id=model_id, dir_json=dir_json)
        else:
            auto_evaluator.setup_auto_kfold_cv_eval(debug=debug, model_id=model_id, data_id=data_id,
                                                    dir_data=dir_data, dir_output=dir_output)
        global_study.optimize(auto_evaluator, n_trials=10)

        global_study_dic = {}
        global_study_dic["best_params"] = global_study.best_params
        global_study_dic["best_value"] = global_study.best_value
        dir_best = '/'.join([dir_json, 'best_json'])
        if not os.path.exists(dir_best): os.makedirs(dir_best)
        with open('/'.join([dir_best, 'Best_{}Parameter.json'.format(model_id)]), 'w') as f:
            json.dump(global_study_dic, f, indent=4)


