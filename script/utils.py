# -*- coding: utf-8 -*-

import time

import numpy as np
import pandas as pd
from sklearn.externals import joblib

from const import *


def timer(func):
	def _wrapper(*args, **kwargs):
		begin = time.time()

		ret = func(*args, **kwargs)

		print '*' * 20
		print '{} cost time: {:.2f} minute'.format(func.__name__, (time.time()-begin)/60)
		print '*' * 30

		return ret

	return _wrapper


def concat(dfs, axis=0):
	"""
	:param dfs:
	:param axis: 0 by row, 1 by column
	:return:
	"""
	return pd.concat(dfs, axis=axis, ignore_index=True, copy=False)


def preprocess_reorder_product_data(df):
	# corr_path = INPUT_DIR+'corr.csv'
	# corr_path = INPUT_DIR+'corr_ratio.csv'
	# corr_path = INPUT_DIR+'corr_interaction_rank.csv'
	# corr_path = INPUT_DIR+'corr_diff.csv'
	# corr_path = INPUT_DIR+'corr_std.csv'
	# corr_path = INPUT_DIR+'corr_std.csv'
	corr_path = INPUT_DIR+'corr_diff_order_streaks_fix_add_new.csv'

	label_name = 'label'
	fill_nan = False

	# force_drop_cols = [UID, PID, AID, DID]
	no_use_cols = [
		'product_product_count',
		'aisle_order_number_max',
		'department_order_number_max',
		# 'department_bought_times'
	]

	less_important_cols = [
		'user_aisle_order_dow_mean',
		'user_product_add_to_cart_order_max',
		'user_aisle_order_dow_mean',
		'user_department_add_to_cart_order_min',
		'user_department_order_number_min',
		'user_aisle_add_to_cart_order_max',
		'user_aisle_add_to_cart_order_min',
		'user_product/user_department_add_to_cart_order_min',
		'user_department_add_to_cart_order_min/user_order_size_mean',
		'product_order_number_max',
		'department_order_hour_of_day_mean',
		'user_aisle_days_since_prior_order_max',
		'user_aisle_days_since_prior_order_min',
		'user_department_days_since_prior_order_max',
		'user_department_days_since_prior_order_min',
		'user_aisle_bought_times/user_bought_times',

		# 为了保证数据顺序一致，暂时不能加
		# 'user_product/user_department_bought_times',
	]

	force_drop_cols = [UID] + no_use_cols + less_important_cols

	return preprocess_data(df, label_name, corr_path, fill_nan, force_drop_cols)


def preprocess_reorder_size_data(df):
	label_name = 'reorder_size'
	# corr_path = REORDER_INPUT_DIR+'corr_train.csv'
	corr_path = REORDER_INPUT_DIR+'corr_all_reorder.csv'
	fill_nan = False
	no_use_cols = ['days_since_prior_order', 'order_number', 'order_hour_of_day', 'order_dow']
	force_drop_cols = [UID, OID, 'reorder_sizes_min']
	return preprocess_data(df, label_name, corr_path, fill_nan, force_drop_cols)


def preprocess_none_order_data(df):
	label_name = 'label'
	# corr_path = NONE_ORDER_INPUT_DIR+'corr.csv'
	corr_path = NONE_ORDER_INPUT_DIR+'corr_ratio.csv'
	fill_nan = False
	no_use_cols = []
	force_drop_cols = [UID, OID, 'reorder_size_min']
	return preprocess_data(df, label_name, corr_path, fill_nan, force_drop_cols)


def preprocess_data(df, label_name, corr_path, fill_nan, force_drop_cols):
	df_corr = pd.read_csv(corr_path, index_col=0)
	df_corr = abs(df_corr).fillna(0.0)

	for key in df.columns:
		if '/' in str(key):
			# for memory, replace inplace
			df[key].replace([np.inf, -np.inf], np.nan, inplace=True)

	# for estimator that cannot handle nan value
	if fill_nan:
		df.fillna(-999, inplace=True)

	for key in df_corr.columns:
		df_corr.loc[key][key] = 0.0

	keys1, keys2, scores = [], [], []
	for key1 in df_corr.columns:
		if key1 == label_name:
			continue
		for key2 in df_corr.columns:
			if key2 == label_name:
				continue
			keys1.append(key1)
			keys2.append(key2)
			scores.append(df_corr.loc[key1][key2])

	df_feature_corr = pd.DataFrame({'key1': keys1, 'key2': keys2, 'score': scores}).sort_values('score', ascending=False)
	df_corr_mean = df_feature_corr.groupby('key1').mean()
	drop_columns = set()

	for idx, row in df_feature_corr.iterrows():
		if row['score'] > 0.95:
			key1, key2, score = row['key1'], row['key2'], row['score']
			if key1 in drop_columns or key2 in drop_columns:
				continue
			if df_corr_mean.loc[key1]['score'] > df_corr_mean.loc[key2]['score']:
				drop_columns.add(key1)
			else:
				drop_columns.add(key2)

	for key in force_drop_cols:
		drop_columns.add(key)

	print 'preprocess, drop columns num: {}, names:{}'.format(len(drop_columns), drop_columns)
	df = df.drop(list(drop_columns), axis=1)

	return df


def save_model(model, param=None, model_name=None):
	if model_name is not None:
		name = '_'.join([model_name, get_now_str()])
	elif param is not None:
		param = param.items()
		param.sort(key=lambda elem: elem[0])
		param = map(lambda elem: ':'.join([str(elem[0]), str(elem[1])]), param)
		name = '_'.join([model.__class__.__name__] + param + [get_now_str()])
	else:
		name = '_'.join([model.__class__.__name__, get_now_str()])

	name = name + '.pkl'
	# print name

	joblib.dump(model, MODEL_DIR+name)
	return name


def load_model(name):
	return joblib.load(MODEL_DIR+name)


def get_now_str():
	import datetime
	return datetime.datetime.now().strftime("%m-%d-%H-%M-%S")


def get_feature_score(feature_score):
	features = []
	scores = []
	for k, v in feature_score.iteritems():
		features.append(k)
		scores.append(v)
	df_feature = pd.DataFrame({'features': features, 'scores': scores}).sort_values('scores', ascending=False)
	return df_feature.reset_index(drop=True)


def get_lgbm_feature_score(features, scores):
	df_feature = pd.DataFrame({'features': features, 'scores': scores}).sort_values('scores', ascending=False)
	return df_feature.reset_index(drop=True)


def get_feature_all_type_score(f_name, xgb_model):
	f_weight = get_feature_score(xgb_model._Booster.get_fscore())
	f_gain = get_feature_score(xgb_model._Booster.get_score(importance_type='gain'))
	f_cover = get_feature_score(xgb_model._Booster.get_score(importance_type='cover'))
	total = f_weight.shape[0]

	print 'total feature num:', total
	print 'gain'
	feature = f_gain[f_gain.features == f_name]
	if not feature.empty:
		print (feature.index[0] + 1) / float(total)
	print feature
	print 'weight'
	feature = f_weight[f_weight.features == f_name]
	if not feature.empty:
		print (feature.index[0] + 1) / float(total)
	print feature
	print 'cover'
	feature = f_cover[f_cover.features == f_name]
	if not feature.empty:
		print (feature.index[0] + 1) / float(total)
	print feature


def get_feature_all_type_lgbm(f_name, f_gain, f_weight):
	total = f_gain.shape[0]
	print 'total feature num:', total
	print 'gain'
	feature = f_gain[f_gain.features == f_name]
	if not feature.empty:
		print (feature.index[0] + 1) / float(total)
	print feature

	print 'weight'
	feature = f_weight[f_weight.features == f_name]
	if not feature.empty:
		print (feature.index[0] + 1) / float(total)
	print feature


def f1_score_single(y_true, y_pred):
	# print y_true, y_pred
	for y in list(y_true) + list(y_pred):
		assert type(y) is not str

	y_true = set(y_true)
	y_pred = set(y_pred)
	cross_size = len(y_true & y_pred)
	if cross_size == 0:
		return 0.
	p = 1. * cross_size / len(y_pred)
	r = 1. * cross_size / len(y_true)
	# print y_true, y_pred
	# print cross_size, p, r, 2 * p * r / (p + r)
	return 2 * p * r / (p + r)


if __name__ == '__main__':
	pass
	name = 'XGBClassifier_colsample_bytree:0.8_learning_rate:0.1_max_depth:8_n_estimators:150_scale_pos_weight:2_seed:0_subsample:0.8_07-27-02-01-25.pkl'
	xgb_model = load_model(name)
	df_feature_gain = get_feature_score(xgb_model._Booster.get_score(importance_type='gain'))
	print df_feature_gain.head(50)

	# print model.get_params()
	# print get_now_str()
	# print 1988721 / float(4096)
