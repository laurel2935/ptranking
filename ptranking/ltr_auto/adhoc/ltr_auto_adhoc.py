
import os
import json
import optuna

from ptranking.ltr_adhoc.eval.eval_utils import ndcg_at_k
from ptranking.ltr_adhoc.eval.ltr import LTREvaluator
from ptranking.ltr_auto.base.auto_parameter import AutoEvalSetting, AutoDataSetting, AutoScoringFunctionParameter
from ptranking.ltr_auto.adhoc.adhoc_auto_parameter import AutoModelParameter

from ptranking.ltr_auto.adhoc.adhoc_auto_parameter import RankNetAutoParameter, LambdaRankAutoParameter,\
    ApproxNDCGAutoParameter, STListNetAutoParameter, ListMLEAutoParameter, WassRankAutoParameter

class LTRObjective(object):
    """
    The customized optuna objective over training data and validation data
    """
    def __init__(self, ranker=None, epochs=None, vali_k=None, label_type=None, train_data=None, vali_data=None, gpu=False, device=None):
        self.ranker = ranker
        self.epochs = epochs
        self.vali_k = vali_k
        self.train_data = train_data
        self.vali_data = vali_data
        self.label_type = label_type
        self.gpu, self.device = gpu, device

    def __call__(self, trial):
        for epoch_k in range(1, self.epochs + 1):
            # one-epoch fitting over the entire training data
            presort = self.train_data.presort
            for qid, batch_rankings, batch_stds in self.train_data:  # _, [batch, ranking_size, num_features], [batch, ranking_size]
                if self.gpu: batch_rankings, batch_stds = batch_rankings.to(self.device), batch_stds.to(self.device)
                batch_loss, stop_training = self.ranker.train(batch_rankings, batch_stds, qid=qid, epoch_k=epoch_k,
                                                         presort=presort, label_type=self.label_type)
                #print(batch_loss)

                if stop_training: break

            # Report intermediate objective value.
            vali_eval_tmp = ndcg_at_k(ranker=self.ranker, test_data=self.vali_data, k=self.vali_k,
                                      label_type=self.label_type, gpu=self.gpu, device=self.device)
            vali_eval_v = vali_eval_tmp.data.numpy()
            print(vali_eval_v)

            intermediate_value = vali_eval_v
            trial.report(intermediate_value, epoch_k)

            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.TrialPruned()

        # using validation data again
        vali_eval_tmp = ndcg_at_k(ranker=self.ranker, test_data=self.vali_data,
                                  k=self.vali_k, label_type=self.label_type, gpu=self.gpu, device=self.device)
        vali_eval_v = vali_eval_tmp.data.numpy()

        return vali_eval_v


class AutoLTREvaluator(LTREvaluator):
    """
    The class for evaluating different adversarial adversarial ltr methods.
    """
    def __init__(self, frame_id='AutoLTR', cuda=None):
        super(AutoLTREvaluator, self).__init__(frame_id=frame_id, cuda=cuda)

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
            auto_data_eval_sf_json = dir_json + 'Auto_Data_Eval_ScoringFunction.json'
            para_json = dir_json + model_id + "AutoParameter.json"

            self.set_eval_setting(debug=debug, eval_json=auto_data_eval_sf_json)
            self.set_data_setting(data_json=auto_data_eval_sf_json)
            self.set_scoring_function_setting(sf_json=auto_data_eval_sf_json)
            self.set_model_setting(model_id=model_id, para_json=para_json)
        else:
            self.set_eval_setting(debug=debug, dir_output=dir_output)
            self.set_data_setting(debug=debug, data_id=data_id, dir_data=dir_data)
            self.set_scoring_function_setting(debug=debug)
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

    def set_data_setting(self, data_json=None, debug=False, data_id=None, dir_data=None):
        if data_json is not None:
            self.data_setting = AutoDataSetting(data_json=data_json)
        else:
            self.data_setting = AutoDataSetting(debug=debug, data_id=data_id, dir_data=dir_data)

    def set_eval_setting(self, eval_json=None, debug=False, dir_output=None):
        if eval_json is not None:
            self.eval_setting = AutoEvalSetting(debug=debug, eval_json=eval_json)
        else:
            self.eval_setting = AutoEvalSetting(debug=debug, dir_output=dir_output)

    def set_scoring_function_setting(self, sf_json=None, debug=None, data_dict=None):
        if sf_json is not None:
            self.sf_parameter = AutoScoringFunctionParameter(sf_json=sf_json)
        else:
            self.sf_parameter = AutoScoringFunctionParameter(debug=debug)

    def set_model_setting(self, model_id=None, para_json=None, debug=False):
        if model_id in ['RankMSE', 'ListNet', 'RankCosine']: # ModelParameter is sufficient
            self.model_parameter = AutoModelParameter(model_id=model_id)
        else:
            if para_json is not None:
                self.model_parameter = globals()[model_id + "AutoParameter"](para_json=para_json)
            else: # the 3rd type, where debug-mode enables quick test
                self.model_parameter = globals()[model_id + "AutoParameter"](debug=debug)


    def __call__(self, trial):
        # sample params
        sf_para_dict = self.sf_parameter.grid_search(trial)
        if sf_para_dict['id'] == 'ffnns':
            sf_para_dict['ffnns'].update(dict(num_features=self.data_dict['num_features']))

        model_para_dict = self.model_parameter.grid_search(trial)

        k_flod_average = 0.
        for i in range(self.fold_num):  # evaluation over k-fold data
            fold_k = i+1
            study = self.k_studies[i]

            train_data, test_data, vali_data = self.load_data(self.eval_dict, self.data_dict, fold_k)

            ranker = self.load_ranker(model_para_dict=model_para_dict, sf_para_dict=sf_para_dict)
            study.optimize(LTRObjective(ranker=ranker, epochs=self.epochs,  vali_k=self.vali_k,
                                        label_type=train_data.label_type,
                                        train_data=train_data, vali_data=vali_data),
                           n_trials=1) # ??? the meaning of n_trials
            # store loss
            vali_eval_tmp = ndcg_at_k(ranker=ranker, test_data=vali_data, k=self.vali_k,
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

        dir_best = '/'.join([self.dir_run, 'best_json'])
        if not os.path.exists(dir_best): os.makedirs(dir_best)
        with open('/'.join([dir_best, 'Best_{}Parameter.json'.format(model_id)]), 'w') as f:
            json.dump(global_study_dic, f, indent=4)