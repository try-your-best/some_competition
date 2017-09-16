# -*- coding: utf-8 -*-

import time
import threading
import gc
import random

import pandas as pd
import numpy as np
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn import ensemble
from sklearn.datasets import make_classification
import xgboost as xgb

from const import *
import utils

np.random.seed(RAND_SEED)
random.seed(RAND_SEED)


class PositiveTune(object):

	def __init__(self, start, end, step):
		self.start = start
		self.end = end
		self.step = step


def full_f1_score(label, pred):
	precision = precision_score(label, pred)
	recall = recall_score(label, pred)
	f1 = 2.0 * precision * recall / (precision + recall)
	# _f1 = f1_score(label, pred)
	# print 'full_f1_score', f1, _f1
	return f1, precision, recall


def customized_f1_score(df):
	if df.label.any():
		return f1_score(df.label, df.pred) if df.pred.any() else 0
	else:
		return 0 if df.pred.any() else 1


def mean_f1_score(df_result):
	return df_result.groupby(UID).apply(customized_f1_score).mean()


def validate_model(model, X_validate, Y_validate, computeAuc=False, positiveTune=None):
	"""
	:param model:
	:param X_validate:
	:param Y_validate:
	:param computeAuc:
	:param positiveTune: PositiveTune type. set it to None to turn tuning off
	:return:
	"""
	df_result = Y_validate[[UID, 'label']]
	proba = model.predict_proba(X_validate)
	df_result['proba'] = proba[:, 1]

	auc = roc_auc_score(df_result['label'], df_result['proba']) if computeAuc else 0

	f1, precision, recall, proba_threshold, top_num, mean_f1 = 0, 0, 0, 0, 0, 0

	if positiveTune is not None:
		df_result = df_result.sort_values('proba', ascending=False).reset_index(drop=True)
		sample_len = df_result.shape[0]
		df_result['pred'] = np.zeros(sample_len)

		for num in xrange(positiveTune.start, positiveTune.end, positiveTune.step):
			df_result.ix[:num, 'pred'] = 1

			cur_f1, cur_precision, cur_recall = full_f1_score(df_result['label'], df_result['pred'])
			cur_mean_f1 = mean_f1_score(df_result)

			if mean_f1 < cur_mean_f1:
				# if f1 > cur_f1:
				# 	print '!'*10, 'waring, mean_f1 and f1 conflict! mean_f1: {} < cur_mean_f1:{}, f1: {} > cur_f1: {}, ' \
				# 		'cur_precision： {}, cur_recall: {}, cur_top_num: {}'.format(
				# 			mean_f1, cur_mean_f1, f1, cur_f1, cur_precision, cur_recall, num)

				f1 = cur_f1
				precision = cur_precision
				recall = cur_recall
				top_num = num
				proba_threshold = df_result['proba'][top_num-1]
				mean_f1 = cur_mean_f1

	return auc, mean_f1, f1, precision, recall, proba_threshold, top_num


def worker(cls, params, X_train, y_train, X_validate, Y_validate, positiveTune, save_model, lock):
	for param in params:
		begin = time.time()
		model = cls(**param)
		model.fit(X_train, y_train)

		save_model_name = None
		if save_model:
			save_model_name = utils.save_model(model, param)

		proba = model.predict_proba(X_train)
		auc_train = roc_auc_score(y_train, proba[:, 1])
		auc, mean_f1, f1, precision, recall, proba_threshold, top_num = validate_model(
			model, X_validate, Y_validate, True, positiveTune)

		lock.acquire()
		print '*' * 20
		print 'validate result: auc: {}, mean_f1: {}, f1: {}, precision: {}, recall: {}, proba_threshold: {}, top_num: {}'.format(
			auc, mean_f1, f1, precision, recall, proba_threshold, top_num)
		print 'train result: auc: {}'.format(auc_train)
		print 'model parameters:', param
		print 'save model name: {}'.format(save_model_name)
		print 'cost time {} minute'.format((time.time() - begin)/60)
		print '*' * 30
		lock.release()


def train_model(cls, param_grid, X_train, y_train, X_validate, Y_validate, positiveTune, save_model=False, thread_num=1):
	params = list(ParameterGrid(param_grid))

	if len(params) < thread_num:
		thread_num = len(params)

	thread_params = [[] for _ in xrange(thread_num)]

	for idx, param in enumerate(params):
		thread_params[idx % thread_num].append(param)

	random.shuffle(thread_params)

	# print thread_params
	lock = threading.Lock()
	for idx in xrange(thread_num):
		time.sleep(1)
		thread = threading.Thread(target=worker,
			args=(cls, thread_params[idx], X_train, y_train, X_validate, Y_validate, positiveTune, save_model, lock))
		thread.start()
		thread.join()


def test_train():
	run('../input/train_sample.csv', '../input/validate_sample.csv')
	# X, y = make_classification(1000, n_features=5, n_informative=2, n_redundant=2, n_classes=2, random_state=0)
	# df = pd.DataFrame(np.hstack((X, y[:, None])), columns=['col' + str(idx) for idx in range(5)] + ['label'])
	# val_min_index = 800
	# train = df.iloc[:val_min_index]
	# validate = df.iloc[val_min_index:]
	#
	# X_train = train.drop('label', axis=1)
	# y_train = train.label
	# X_validate = validate.drop('label', axis=1)
	# y_validate = validate.label
	#
	# param_grid = dict(
	# 	n_estimators=[10, 20, 30],
	# 	learning_rate=[0.1],
	# 	seed=[0]
	# )
	#
	# cls = xgb.XGBClassifier
	#
	# train_model(cls, param_grid, X_train, y_train, X_validate, y_validate, 2)


def get_train_set(train_path, scale_pos_weight=None):
	df_train = pd.read_csv(train_path, compact_ints=True)
	positive = df_train[df_train.label == 1]
	negative = df_train[df_train.label == 0].sample(positive.shape[0] * scale_pos_weight)
	# negative = df_train[df_train.label == 0]
	scale_pos_weight = negative.shape[0] / float(positive.shape[0])
	print 'sample num: {}, scale_pos_weight: {}'.format(positive.shape[0]+negative.shape[0], scale_pos_weight)

	del df_train
	gc.collect()

	print 'sample train set'
	df_train = utils.concat([positive, negative]).sample(frac=1).reset_index(drop=True)

	df_train = utils.preprocess_reorder_product_data(df_train)
	X_train = df_train.drop(['label'], axis=1)
	y_train = df_train.label

	return X_train, y_train


def get_validate_set(validate_path):
	print 'create validate set'
	df_validate = pd.read_csv(validate_path, compact_ints=True)
	Y_validate = df_validate[[UID, 'label']]
	df_validate = utils.preprocess_reorder_product_data(df_validate)
	X_validate = df_validate.drop(['label'], axis=1)
	return X_validate, Y_validate


def run(train_path, validate_path):
	print 'start run'
	begin = time.time()

	scale_pos_weight = 2
	X_train,  y_train = get_train_set(train_path, scale_pos_weight)
	gc.collect()

	X_validate, Y_validate = get_validate_set(validate_path)
	gc.collect()

	print 'start training'

	save_model = True

	cls = xgb.XGBClassifier
	param_grid = dict(
		n_estimators=[150],
		learning_rate=[0.1],
		subsample=[0.8],
		max_depth=[8],
		colsample_bytree=[0.8],
		# min_child_weight=[3],
		scale_pos_weight=[scale_pos_weight],
		seed=[RAND_SEED],
	)

	# cls = ensemble.RandomForestClassifier
	# param_grid = dict(
	# 	n_estimators=[120, 200],
	# 	max_depth=[12, 13],
	# 	max_features=['sqrt'],
	# 	min_samples_split=[15],
	# 	min_samples_leaf=[10],
	# 	n_jobs=[-1],
	# 	random_state=[0],
	# )

	# cls = ensemble.ExtraTreesClassifier
	# param_grid = dict(
	# 	n_estimators=[200],
	# 	max_depth=[13],
	# 	max_features=['sqrt'],
	# 	min_samples_split=[15],
	# 	min_samples_leaf=[10],
	# 	n_jobs=[-1],
	# 	random_state=[0],
	# )

	# cls = ensemble.GradientBoostingClassifier
	# param_grid = dict(
	# 	n_estimators=[150],
	# 	learning_rate=[0.1],
	# 	subsample=[0.8],
	# 	max_depth=[8],
	# 	max_features=[0.1],
	# 	min_samples_split=[500],
	# 	min_samples_leaf=[200],
	# 	random_state=[RAND_SEED],
	# )

	# start = int(X_validate.shape[0] * 0.098)
	# end = int(X_validate.shape[0] * 0.2)
	# positiveTune = PositiveTune(start, end, (end - start) / 5)

	positiveTune = None

	# print 'X_train: {}, X_validate: {}'.format(X_train.shape, X_validate.shape)

	train_model(cls, param_grid, X_train, y_train, X_validate, Y_validate, positiveTune, save_model, 4)

	print 'end!!! cost time: {} minute'.format((time.time()-begin)/60)


def run_cv(train_path, validate_path):
	pass
	scale_pos_weight = 2
	X_train,  y_train = get_train_set(train_path, scale_pos_weight)
	gc.collect()

	X_validate, Y_validate = get_validate_set(validate_path)
	gc.collect()

	cls = xgb.XGBClassifier
	param_grid = dict(
		n_estimators=[150],
		learning_rate=[0.1],
		subsample=[0.8],
		max_depth=[8],
		colsample_bytree=[0.8],
		# min_child_weight=[3],
		scale_pos_weight=[scale_pos_weight],
		seed=[RAND_SEED],
	)

	param = dict(eta=0.03, max_depth=4, subsample=0.9, colsample_bytree=0.85,
	             # scale_pos_weight=scale_pos_weight,
	             objective='binary:logistic', silent=1)




def test_run():
	# run(INPUT_DIR+'train_sample.csv', INPUT_DIR+'validate_sample.csv')
	run(INPUT_DIR+'train_sample_ratio.csv', INPUT_DIR+'validate_sample_ratio.csv')


def formal_run():
	# run(INPUT_DIR+'train.csv', INPUT_DIR+'validate.csv')
	# run(INPUT_DIR+'train_ratio.csv', INPUT_DIR+'validate_ratio.csv')
	# run(INPUT_DIR+'train_interaction_rank.csv', INPUT_DI
	#
	#
	# R+'validate_interaction_rank.csv')
	# run(INPUT_DIR+'train_std.csv', INPUT_DIR+'validate_std.csv')
	run(INPUT_DIR+'train_order_streaks.csv', INPUT_DIR+'validate_order_streaks.csv')



@utils.timer
def tune_positive():
	name = 'Stacker:LogisticRegression_07-23-20-40-05.pkl'

	model = utils.load_model(name)

	validate_path = INPUT_DIR+'validate_diff.csv'
	df_validate = pd.read_csv(validate_path)
	X_validate = utils.preprocess_reorder_product_data(df_validate).drop(['label'], axis=1)
	Y_validate = df_validate[[UID, 'label']]

	# del df_validate
	# gc.collect()

	start = int(X_validate.shape[0] * 0.098)
	end = int(X_validate.shape[0] * 0.2)
	positiveTune = PositiveTune(start, end,  (end - start) / 50)

	auc, mean_f1, f1, precision, recall, proba_threshold, top_num = validate_model(
		model, X_validate, Y_validate, False, positiveTune)

	print 'validate result: auc: {}, mean_f1: {}, f1: {}, precision: {}, recall: {}, proba_threshold: {}, top_num: {}'.format(
		auc, mean_f1, f1, precision, recall, proba_threshold, top_num)


def run_tune_positive():
	# name = 'XGBClassifier_colsample_bytree:0.8_learning_rate:0.1_max_depth:8_n_estimators:150_seed:0_subsample:0.8.plk'
	# name = 'XGBClassifier_colsample_bytree:0.8_learning_rate:0.1_max_depth:8_n_estimators:150_seed:0_subsample:0.8_07-17-00-40-20.plk'
	# name = 'XGBClassifier_colsample_bytree:0.8_learning_rate:0.1_max_depth:8_n_estimators:150_scale_pos_weight:2_seed:0_subsample:0.8_07-18-21-51-19.pkl'
	# name = 'XGBClassifier_colsample_bytree:0.8_learning_rate:0.1_max_depth:8_n_estimators:150_seed:0_subsample:0.8_07-20-23-56-07.pkl'
	# name = 'GradientBoostingClassifier_learning_rate:0.1_max_depth:8_max_features:0.1_min_samples_leaf:200_min_samples_split:500_n_estimators:150_random_state:0_subsample:0.8_07-22-00-58-23.pkl'
	# name = 'XGBClassifier_colsample_bytree:0.8_learning_rate:0.1_max_depth:10_n_estimators:150_scale_pos_weight:5.0_seed:0_subsample:0.8_07-22-14-05-55.pkl'
	name = 'XGBClassifier_colsample_bytree:0.8_learning_rate:0.1_max_depth:8_n_estimators:150_scale_pos_weight:5.0_seed:0_subsample:0.8_07-22-18-22-57.pkl'
	tune_positive()

if __name__ == '__main__':
	pass
	# test_train()
	# test_run()
	formal_run()
	# tune_positive()
