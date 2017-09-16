# -*- coding: utf-8 -*-

import time
import threading
import gc
import random

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn import ensemble, model_selection, calibration
from sklearn.datasets import make_classification
import xgboost as xgb
import lightgbm as lgb

from const import *
import utils

np.random.seed(RAND_SEED)
random.seed(RAND_SEED)


def get_train_and_validate_data(train_file, validate_file, negative_ratio=None):
	df_train = pd.read_csv(train_file, compact_ints=True)

	# if negative_ratio:
	# 	positive = df_train[df_train.label == 1]
	# 	negative = df_train[df_train.label == 0].sample(positive.shape[0] * negative_ratio)
	# 	df_train = utils.concat([positive, negative]).sample(frac=1).reset_index(drop=True)

	df_train = utils.preprocess_reorder_product_data(df_train)
	X_train = df_train.drop(['label'], axis=1)
	y_train = df_train['label']

	del df_train
	gc.collect()

	df_validate = pd.read_csv(validate_file, compact_ints=True)
	Y_validate = df_validate[[UID, 'label']]

	df_validate = utils.preprocess_reorder_product_data(df_validate)
	X_validate = df_validate.drop(['label'], axis=1)

	del df_validate
	gc.collect()
	return X_train, y_train, X_validate, Y_validate


def get_train_and_validate_data_from_cache(train_file, validate_file, only_validate=False, is_concat=False):
	print 'get_data train_file:{}, validate_file:{}, only_validate: {}, concat: {}'.format(
		train_file, validate_file, only_validate, is_concat)

	dtypes = dict(
		label=np.float32,
	)
	embedings = list(range(32))
	for col in embedings:
		dtypes[col] = np.float32

	useless_cols = [
		'department_product_count',
		'department_order_count',
		'department_order_dow_mean',
		'department_days_since_prior_order_mean',
		'department_add_to_cart_order_mean',
		'department_bought_times',
		'department_reorder_ratio',

		'aisle_add_to_cart_order_mean',
		'aisle_product_count',
		'aisle_days_since_prior_order_mean',
		'aisle_order_hour_of_day_mean',
		'aisle_order_dow_mean',
		'aisle_bought_times',
		'aisle_reorder_ratio',
	]

	if not only_validate or is_concat:
		df_train = pd.read_csv(train_file, compact_ints=True, dtype=dtypes)
		y_train = df_train['label']
		df_train.drop(['label']+useless_cols, axis=1, inplace=True)
	else:
		y_train = None
		df_train = None

	# df_validate, y_validate = None, None
	df_validate = pd.read_csv(validate_file, compact_ints=True, dtype=dtypes)
	y_validate = df_validate['label']
	df_validate.drop([UID, 'label']+useless_cols, axis=1, inplace=True)

	if is_concat:
		df_train = utils.concat([df_train, df_validate])
		df_validate = None
		y_train = y_train.append(y_validate, ignore_index=True)
		y_validate = None
		gc.collect()

	return df_train, y_train, df_validate, y_validate


def get_predict_file(input_file, is_test_set=False, is_train_set=False):
	print 'get_predict_file', is_test_set, is_train_set
	if is_test_set:
		dtypes = dict()
	else:
		dtypes = dict(
			label=np.float32,
			# product_id=np.uint16,
			# aisle_id=np.uint8,
			# department_id=np.uint8,
		)

	embedings = list(range(32))
	for col in embedings:
		dtypes[col] = np.float32

	useless_cols = [
		'department_product_count',
		'department_order_count',
		'department_order_dow_mean',
		'department_days_since_prior_order_mean',
		'department_add_to_cart_order_mean',
		'department_bought_times',
		'department_reorder_ratio',

		'aisle_add_to_cart_order_mean',
		'aisle_product_count',
		'aisle_days_since_prior_order_mean',
		'aisle_order_hour_of_day_mean',
		'aisle_order_dow_mean',
		'aisle_bought_times',
		'aisle_reorder_ratio',
	]

	df_predict = pd.read_csv(input_file, compact_ints=True, dtype=dtypes)

	if is_train_set:
		df_result = None
		df_predict.drop(['label'] + useless_cols, axis=1, inplace=True)
	else:
		if LABEL in df_predict.columns:
			df_result = df_predict[[UID, PID, LABEL]]
		else:
			df_result = df_predict[[UID, PID]]

		df_result[PID] = df_result[PID].astype(np.int32)

		if LABEL in df_predict.columns:
			df_predict.drop([UID, 'label'] + useless_cols, axis=1, inplace=True)
		else:
			df_predict.drop([UID] + useless_cols, axis=1, inplace=True)

	return df_predict, df_result


def get_key_param(params):
	ignore = set(['task', 'boosting_type', 'objective', 'metric', 'verbose'])
	key_param = {}
	for k, v in params.iteritems():
		if k not in ignore:
			key_param[k] = v

	return key_param


@utils.timer
def save_train_and_validate_for_speed(train_file, validate_file):
	print 'start save ', train_file, validate_file
	df_train = pd.read_csv(train_file)
	df_train = utils.preprocess_reorder_product_data(df_train)
	df_train.to_csv(train_file+'_core.csv', index=False)
	del df_train
	gc.collect()

	df_validate = pd.read_csv(validate_file)
	users = df_validate[UID]
	df_validate = utils.preprocess_reorder_product_data(df_validate)
	df_validate[UID] = users
	df_validate.to_csv(validate_file+'_core.csv', index=False)


@utils.timer
def save_used_feature_for_speed(input_file):
	df = pd.read_csv(input_file)
	users = df[UID]
	df = utils.preprocess_reorder_product_data(df)
	df[UID] = users
	df.to_csv(input_file+'_core.csv', index=False)


def run_fix():
	df1 = pd.read_csv(INPUT_DIR + 'test_diff_order_streaks_fix_add_new.csv')
	df2 = pd.read_csv(INPUT_DIR + 'test_diff_order_streaks_fix_add_new_core.csv')
	df2[UID] = df1[UID]
	df2.to_csv(INPUT_DIR + 'test_diff_order_streaks_fix_add_new_core.csv', index=False)


@utils.timer
def save_lgbm_dataset(train_file, validate_file):
	X_train, y_train, X_validate, y_validate = get_train_and_validate_data_from_cache(train_file, validate_file)
	categories = [PID, AID, DID]
	lgb_train = lgb.Dataset(X_train, y_train, categorical_feature=categories)
	del X_train, y_train
	gc.collect()
	lgb_train.save_binary(train_file+'.bin')

	lgb_eval = lgb.Dataset(X_validate, y_validate, reference=lgb_train, categorical_feature=categories)
	lgb_eval.save_binary(validate_file + '.bin')
	# del X_validate
	# gc.collect()


@utils.timer
def save_xgb_dataset(train_file, validate_file):
	X_train, y_train, X_validate, y_validate = get_train_and_validate_data_from_cache(train_file, validate_file)
	categories = [PID, AID, DID]
	X_train.drop(categories, axis=1, inplace=True)
	xgb_train = xgb.DMatrix(X_train, y_train)
	del X_train, y_train
	gc.collect()
	xgb_train.save_binary(train_file+'_xgb.bin')
	# lgb_eval = lgb.Dataset(X_validate, y_validate, reference=lgb_train, categorical_feature=categories)
	# lgb_eval.save_binary(validate_file + '_xgb.bin')


@utils.timer
def run_cv(train_file, validate_file, is_save_model=True):
	begin = time.time()
	# X_train, y_train, X_validate, Y_validate = get_train_and_validate_data(train_file, validate_file)
	# X_train, y_train, X_validate, Y_validate = get_train_and_validate_data_from_cache(train_file, validate_file)
	#
	# categories = [PID, AID, DID]
	# lgb_train = lgb.Dataset(X_train, y_train, categorical_feature=categories)
	# print 'train shape', X_train.shape
	# del X_train, y_train
	# gc.collect()
	#
	# lgb_eval = lgb.Dataset(X_validate, Y_validate['label'], reference=lgb_train, categorical_feature=categories)
	# del X_validate
	# gc.collect()
	lgb_train = lgb.Dataset(train_file)
	lgb_eval = lgb.Dataset(validate_file)

	print 'create data cost: {} minute'.format((time.time() - begin) / 60)

	params = dict(
		task='train',
		boosting_type='gbdt',
		objective='binary',
		metric={'binary_logloss', 'auc'},
		num_leaves=170,
		max_depth=8,
		learning_rate=0.04,
		feature_fraction=0.7,
		bagging_fraction=0.85,
		bagging_freq=3,
		verbose=3,
	)

	print 'start train'
	gbm = lgb.train(
		params,
		lgb_train,
		num_boost_round=586,
		valid_sets=[lgb_eval, lgb_train],
		valid_names=['val_set', 'train_set'],
		# early_stopping_rounds=30
	)
	if is_save_model:
		# print 'key params: ', get_key_param(params)
		save_model_name = utils.save_model(gbm, get_key_param(params))
		print 'save model name: ', save_model_name


@utils.timer
def save_user_ids_for_cv(input_file):
	df = pd.read_csv(input_file)
	df[UID].to_pickle(input_file+'_users.pkl')


@utils.timer
def run_full_cv(train_file, user_file):
	lgb_train = lgb.Dataset(train_file)
	lgb_train.construct()
	users = pd.read_pickle(user_file)

	assert lgb_train.num_data() == users.shape[0]

	kf = model_selection.GroupKFold(n_splits=5).split(users, users, users)

	params = dict(
		task='train',
		boosting_type='gbdt',
		objective='binary',
		num_leaves=170,
		max_depth=8,
		learning_rate=0.03,
		feature_fraction=0.6,
		bagging_fraction=0.85,
		bagging_freq=3,
		verbose=3,
	)

	lgb.cv(params, lgb_train,
		metrics=['binary_logloss', 'auc'],
		num_boost_round=2000,
		early_stopping_rounds=30,
		folds=kf,
		verbose_eval=1
	)


# @utils.timer
# def run_calibration(train_file, validate_file, is_save_model=True):
# 	print 'run_calibration', train_file, validate_file, is_save_model
# 	begin = time.time()
# 	X_train, y_train, X_validate, y_validate = get_train_and_validate_data_from_cache(train_file, validate_file)
# 	print 'train set {}, {}, val set {}, {}'.format(X_train.shape, y_train.shape, X_validate.shape, y_validate.shape)
# 	print 'create data cost: {} minute'.format((time.time() - begin) / 60)
#
# 	params = dict(
# 		objective='binary',
# 		num_leaves=170,
# 		max_depth=8,
# 		n_estimators=586,
# 		learning_rate=0.04,
# 		colsample_bytree=0.7,
# 		subsample=0.85,
# 		subsample_freq=3,
# 	)
# 	est = lgb.LGBMClassifier(seed=RAND_SEED, **params)
# 	est.fit(X_train, y_train,
# 		eval_metric=['binary_logloss', 'auc'],
# 		eval_set=[(X_train, y_train), (X_validate, y_validate)],
# 		eval_names=['train', 'val'],
# 		categorical_feature=[PID, AID, DID],
# 		verbose=True,
# 	)
# 	if is_save_model:
# 		save_model_name = utils.save_model(est, get_key_param(params))
# 		print 'save model name: ', save_model_name


@utils.timer
def run_train_for_submit(train_file, validate_file, is_save_model=True):
	print 'run_train_for_submit', train_file, validate_file
	begin = time.time()
	X_train, y_train, _, _ = get_train_and_validate_data_from_cache(train_file, validate_file, is_concat=True)
	gc.collect()
	categories = [PID, AID, DID]
	lgb_train = lgb.Dataset(X_train, y_train, categorical_feature=categories)

	del X_train, y_train
	gc.collect()

	print 'create data cost: {} minute'.format((time.time() - begin) / 60)

	params = dict(
		task='train',
		boosting_type='gbdt',
		objective='binary',
		metric={'binary_logloss', 'auc'},
		num_leaves=170,
		max_depth=8,
		learning_rate=0.03,
		feature_fraction=0.6,
		bagging_fraction=0.85,
		bagging_freq=3,
		verbose=3,
	)

	print 'start train'
	gbm = lgb.train(
		params,
		lgb_train,
		num_boost_round=746,
	)

	if is_save_model:
		save_model_name = utils.save_model(gbm, get_key_param(params))
		print 'save model name: ', save_model_name


if __name__ == '__main__':
	pass
	# run_cv(
	# 	INPUT_DIR+'train_diff_order_streaks_fix_add_new.csv',
	# 	INPUT_DIR + 'validate_diff_order_streaks_fix_add_new.csv',
	# )

	# save_train_and_validate_for_speed(
	# 	INPUT_DIR+'train_diff_order_streaks_fix_add_new.csv',
	# 	INPUT_DIR + 'validate_diff_order_streaks_fix_add_new.csv',
	# )
	#
	# run_cv(
	# 	INPUT_DIR+'train_diff_order_streaks_fix_add_new_core.csv',
	# 	INPUT_DIR + 'validate_diff_order_streaks_fix_add_new_core.csv',
	# )

	# run_train_for_submit(
	# 	INPUT_DIR + 'train_diff_order_streaks_fix_add_new_core.csv',
	# 	INPUT_DIR + 'validate_diff_order_streaks_fix_add_new_core.csv',
	# )


	# save_lgbm_dataset(
	# 	INPUT_DIR + 'train_diff_order_streaks_fix_add_new_core.csv',
	# 	INPUT_DIR + 'validate_diff_order_streaks_fix_add_new_core.csv',
	# )

	# save_used_feature_for_speed(
	# 	INPUT_DIR+'test_diff_order_streaks_fix_add_new.csv',
	# 	INPUT_DIR+'test_diff_order_streaks_fix_add_new_core.csv',
	# )

	# run_fix()

	# save_train_and_validate_for_speed(
	# 	INPUT_DIR+'train_diff_order_streaks_fix2_add_new.csv',
	# 	INPUT_DIR + 'validate_diff_order_streaks_fix2_add_new.csv',
	# )

	# save_lgbm_dataset(
	# 	INPUT_DIR + 'train_diff_order_streaks_fix2_add_new.csv_core.csv',
	# 	INPUT_DIR + 'validate_diff_order_streaks_fix2_add_new.csv_core.csv',
	# )

	# save_user_ids_for_cv(INPUT_DIR+'train_diff_order_streaks_fix2_add_new.csv')

	# run_full_cv(
	# 	INPUT_DIR+'train_diff_order_streaks_fix2_add_new.csv_core.csv.bin',
	# 	INPUT_DIR+'train_diff_order_streaks_fix2_add_new.csv_users.pkl'
	# )

	# run_cv(
	# 	INPUT_DIR+'train_diff_order_streaks_fix2_add_new.csv_core.csv.bin',
	# 	INPUT_DIR + 'validate_diff_order_streaks_fix2_add_new.csv_core.csv.bin',
	# )

	# save_xgb_dataset(
	# 	INPUT_DIR + 'train_diff_order_streaks_fix2_add_new.csv_core.csv',
	# 	INPUT_DIR + 'validate_diff_order_streaks_fix2_add_new.csv_core.csv',
	# )

	# run_full_cv(
	# 	INPUT_DIR+'train_diff_order_streaks_fix2_add_new.csv_core.csv.bin',
	# 	INPUT_DIR+'train_diff_order_streaks_fix2_add_new.csv_users.pkl'
	# )

	run_train_for_submit(
		INPUT_DIR + 'train_diff_order_streaks_fix2_add_new.csv_core.csv',
		INPUT_DIR + 'validate_diff_order_streaks_fix2_add_new.csv_core.csv',
	)

	# save_used_feature_for_speed(
	# 	INPUT_DIR+'test_diff_order_streaks_fix2_add_new.csv',
	# )

	# run_calibration(
	# 	INPUT_DIR + 'train_diff_order_streaks_fix2_add_new.csv_core.csv',
	# 	INPUT_DIR + 'validate_diff_order_streaks_fix2_add_new.csv_core.csv',
	# )