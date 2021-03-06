
import json
from itertools import product

from ptranking.data.data_utils import get_default_scaler_setting, get_data_meta
from ptranking.ltr_adhoc.eval.parameter import EvalSetting, DataSetting, ScoringFunctionParameter

class AutoEvalSetting(EvalSetting):
	"""
	Class object for evaluation settings w.r.t. using optuna.
	"""
	def __init__(self, debug=False, dir_output=None, eval_json=None):
		super(AutoEvalSetting, self).__init__(debug=debug, dir_output=dir_output, eval_json=eval_json)

	def load_para_json(self, para_json):
		with open(para_json) as json_file:
			json_dict = json.load(json_file)["AutoEvalSetting"]
		return json_dict

	def to_eval_setting_string(self, log=False):
		"""
		String identifier of eval-setting
		:param log:
		:return:
		"""
		eval_dict = self.eval_dict
		s1, s2 = (':', '\n') if log else ('_', '_')

		vali_obj, epochs = eval_dict['vali_obj'], eval_dict['epochs']

		eval_string = s2.join([s1.join(['epochs', str(epochs)]), s1.join(['Vali_Obj', str(vali_obj)])]) if log \
			else s1.join(['EP', str(epochs), 'Vali_Obj', str(vali_obj)])

		return eval_string

	def load_setting(self):
		"""
		A default setting for evaluation
		:param debug:
		:param data_id:
		:param dir_output:
		:return:
		"""
		if self.use_json: # using json file
			dir_output = self.json_dict['dir_output']
			epochs = 5 if self.debug else self.json_dict['epochs']  # debug is added for a quick check
			vali_obj, vali_k = self.json_dict['vali_obj'], self.json_dict['vali_k']
			cutoffs = self.json_dict['cutoffs']
			do_log, log_step = self.json_dict['do_log'], self.json_dict['log_step']
			do_summary = self.json_dict['do_summary']
			#loss_guided = self.json_dict['loss_guided']
			#mask_label = self.json_dict['mask']['mask_label']
			#choice_mask_type = self.json_dict['mask']['mask_type']
			#choice_mask_ratio = self.json_dict['mask']['mask_ratio']

			base_dict = dict(debug=False, grid_search=True, dir_output=dir_output)
		else:
			base_dict = dict(debug=self.debug, grid_search=True, dir_output=self.dir_output)
			epochs = 20 if self.debug else 100
			vali_obj = False if self.debug else True  # True, False
			vali_k, cutoffs = 5, [1, 3, 5, 10, 20, 50]
			do_log = False if self.debug else True
			log_step = 2
			do_summary, loss_guided = False, False

			#mask_label = False if self.debug else False
			#choice_mask_type = ['rand_mask_all']
			#choice_mask_ratio = [0.2]

		do_validation = True if vali_obj else False
		self.eval_dict = dict(epochs=epochs, vali_obj=vali_obj, vali_k=vali_k, cutoffs=cutoffs,
							  do_log=do_log, log_step=log_step, do_summary=do_summary, do_validation=do_validation)
		self.eval_dict.update(base_dict)

		''' setting for exploring the impact of randomly removing some ground-truth labels '''
		mask_label = False
		mask_type = 'rand_mask_all'
		mask_ratio = 0.2

		mask_dict = dict(mask_label=mask_label, mask_type=mask_type, mask_ratio=mask_ratio)
		self.eval_dict.update(mask_dict)

		return self.eval_dict


class AutoDataSetting(DataSetting):
	"""
	Class object for data settings w.r.t. data loading and pre-process.
	"""
	def __init__(self, debug=False, data_id=None, dir_data=None, data_json=None):
		super(AutoDataSetting, self).__init__(debug=debug, data_id=data_id, dir_data=dir_data, data_json=data_json)

	def load_para_json(self, para_json):
		with open(para_json) as json_file:
			json_dict = json.load(json_file)["AutoDataSetting"]
		return json_dict

	def load_setting(self):
		if self.use_json:
			choice_min_docs = self.json_dict['min_docs']
			choice_min_rele = self.json_dict['min_rele']
			choice_binary_rele = self.json_dict['binary_rele']
			choice_unknown_as_zero = self.json_dict['unknown_as_zero']
			choice_train_presort = self.json_dict['train_presort']
			choice_train_batch_size = self.json_dict['train_batch_size']
			# hard-coding for rarely changed settings
			base_data_dict = dict(data_id=self.data_id, dir_data=self.json_dict["dir_data"], test_presort=True,
								  validation_presort=True, validation_batch_size=1, test_batch_size=1)
		else:
			choice_min_docs = [10]
			choice_min_rele = [1]
			choice_binary_rele = [False]
			choice_unknown_as_zero = [False]
			choice_train_presort = [True]
			choice_train_batch_size = [1]  # number of sample rankings per query

			base_data_dict = dict(data_id=self.data_id, dir_data=self.dir_data, test_presort=True,
								  validation_presort=True, validation_batch_size=1, test_batch_size=1)

		data_meta = get_data_meta(data_id=self.data_id)  # add meta-information
		base_data_dict.update(data_meta)

		choice_scale_data, choice_scaler_id, choice_scaler_level = get_default_scaler_setting(data_id=self.data_id,
																							  grid_search=True)

		for min_docs, min_rele, train_batch_size in product(choice_min_docs, choice_min_rele, choice_train_batch_size):
			threshold_dict = dict(min_docs=min_docs, min_rele=min_rele, train_batch_size=train_batch_size)

			for binary_rele, unknown_as_zero, train_presort in product(choice_binary_rele, choice_unknown_as_zero,
																	   choice_train_presort):
				custom_dict = dict(binary_rele=binary_rele, unknown_as_zero=unknown_as_zero,
								   train_presort=train_presort)

				for scale_data, scaler_id, scaler_level in product(choice_scale_data, choice_scaler_id,
																   choice_scaler_level):
					scale_dict = dict(scale_data=scale_data, scaler_id=scaler_id, scaler_level=scaler_level)

					self.data_dict = dict()
					self.data_dict.update(base_data_dict)
					self.data_dict.update(threshold_dict)
					self.data_dict.update(custom_dict)
					self.data_dict.update(scale_dict)
					return self.data_dict


class AutoScoringFunctionParameter(ScoringFunctionParameter):
	"""
	The parameter class w.r.t. a neural scoring fuction
	"""
	def __init__(self, debug=False, data_dict=None, sf_json=None):
		super(AutoScoringFunctionParameter, self).__init__(debug=debug, data_dict=data_dict, sf_json=sf_json)

	def default_para_dict(self):
		raise NotImplementedError

	def grid_search(self, trial=None):
		"""
		Iterator of hyper-parameters of the stump neural scoring function by sampling based on trial
		:param debug:
		:param data_dict:
		:return:
		"""
		#FBN = False if data_dict['scale_data'] else True  # for feature normalization
		FBN = False
		if self.use_json:
			choice_BN = self.json_dict['BN']
			choice_RD = self.json_dict['RD']
			choice_layers = self.json_dict['layers']
			choice_apply_tl_af = self.json_dict['apply_tl_af']
			choice_hd_hn_tl_af = self.json_dict['hd_hn_tl_af']
		else:
			choice_BN = [False] if self.debug else [True]  # True, False
			choice_RD = [False] if self.debug else [False]  # True, False
			choice_layers = [3] if self.debug else [5]  # 1, 2, 3, 4
			choice_hd_hn_tl_af = ['R', 'CE'] if self.debug else ['R', 'CE', 'S']  # ['R', 'LR', 'RR', 'E', 'SE', 'CE', 'S']
			choice_apply_tl_af = [True]  # True, False

		# fixed and sampled settings
		hd_hn_tl_af = trial.suggest_categorical('hd_hn_tl_af', choice_hd_hn_tl_af)

		# TODO change hard-coding [True] w.r.t categorical
		ffnns_para_dict = dict(FBN=FBN, # essentially corresponds to batch-normalization for query-level features
							   BN=True, # batch normalization for intermediate features
							   num_layers= trial.suggest_int('num_layers', choice_layers[0], choice_layers[-1]), # low, high
							   HD_AF=hd_hn_tl_af,
							   HN_AF=hd_hn_tl_af,
							   TL_AF=hd_hn_tl_af,
							   RD = trial.suggest_categorical('RD', [False]),
							   apply_tl_af = trial.suggest_categorical('apply_tl_af', [True])
							   )
		sf_para_dict = dict()
		sf_para_dict['id'] = self.model_id
		sf_para_dict[self.model_id] = ffnns_para_dict

		self.sf_para_dict = sf_para_dict
		return sf_para_dict
