# -*- coding: utf-8 -*-

from pandas import DataFrame, Series
import pandas as pd
import numpy as np
import utils
from const import *
import gc
import tensorflow as tf

# a = DataFrame({'a': [1, 2]})
#
# def fun(row):
# 	return DataFrame({'c':[1, 2], 'd':[3, 4]})
# 	# return Series([5, 6], index=['c', 'd'])



# a.apply(fun, axis=1)

# print '{:.2f}haha'.format(1.23456)

# df_validate = pd.read_csv('../input/validate.csv')
# print df_validate.columns.size
# print df_validate.info()

import os
import multiprocessing
from multiprocessing import Pool

# def func(i):
#     global a
#     p = multiprocessing.current_process()
#     print i, p.name, p._identity, p.pid, os.getpid(), os.getppid(),  b,  a
#     a += 10
#
# a, b = 1, 1
# pool = Pool(2)
#
# a, b = 2, 2
# print '--------'
# pool.map(func, range(4))
#
# pool = Pool(2)
# print '--------'
# pool.map(func, range(4))
#
# def func(i, c=[]):
#     p = multiprocessing.current_process()
#     print i, p.name, p._identity, p.pid, os.getpid(), c
#     c.append(i)
#
# print '--------'
# pool.map(func,  range(4))
#
# pool = Pool(2)
# print '--------'
# pool.map(func,   range(4))


def sample_data():
	df_train = pd.read_csv('../input/train.csv')
	df_train = df_train[df_train[UID] < 100]
	df_train.to_csv('../input/train_sample.csv', index=False)

	df_validate = pd.read_csv('../input/validate.csv')
	df_validate = df_validate.head(3000)
	df_validate.to_csv('../input/validate_sample.csv', index=False)


def check_positive_step():
	val_len = 1693004
	# print int(val_len * 0.098), int(val_len * 0.2), 207914 / float(val_len)
	POSITIVE_MIN = int(val_len * 0.098)
	POSITIVE_MAX = int(val_len * 0.2)
	print POSITIVE_MIN, POSITIVE_MAX, POSITIVE_MAX - POSITIVE_MIN, (POSITIVE_MAX - POSITIVE_MIN) / 5
	step = (POSITIVE_MAX - POSITIVE_MIN) / 5
	for num in xrange(POSITIVE_MIN, POSITIVE_MAX, step):
		print num


def positive_pct(proba_threshold):
	df = pd.read_csv('../prediction/test_ratio.csv')
	num1 = (df.proba >= proba_threshold).sum()
	num2 = float(df.shape[0])
	print num1, num2, num1 / num2


def test_user():
	df_order = pd.read_csv(ORDER_PATH)
	df1 = df_order.ix[df_order.eval_set == 'train', UID]
	df2 = df_order.ix[df_order.eval_set == 'test', UID]

	print (df1.isin(df2)).any()


def test_StratifiedKFold():
	from sklearn.model_selection import StratifiedKFold
	# X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
	# y = np.array([0, 0, 1, 1])
	X = pd.DataFrame({'a': [1, 2, 3, 4], 'b': [1, 2, 3, 4]})
	y = pd.Series([0, 0, 1, 1])
	skf = StratifiedKFold(n_splits=2, shuffle=True)
	# skf.get_n_splits(X, y)
	print skf.split(X, y), type(skf.split(X, y))
	for i, (train_index, test_index) in enumerate(skf.split(X, y)):
		print i, "TRAIN:", train_index, "TEST:", test_index
		X_train, X_test = X.iloc[train_index], X.iloc[test_index]
		y_train, y_test = y.iloc[train_index], y.iloc[test_index]
		print X_train

from ensemble import EnsembleClassifier

def test_ensemble():
	from sklearn.datasets import make_classification
	from sklearn.metrics import roc_auc_score

	X, y = make_classification(1000, n_features=5, n_informative=2, n_redundant=2, n_classes=2, random_state=0)
	df = pd.DataFrame(np.hstack((X, y[:, None])), columns=['col' + str(idx) for idx in range(5)] + ['label'])
	val_min_index = 800
	train = df.iloc[:val_min_index]
	validate = df.iloc[val_min_index:]

	X_train = train.drop('label', axis=1)
	y_train = train.label
	X_validate = validate.drop('label', axis=1)
	y_validate = validate.label

	name = 'EnsembleClassifier_07-23-00-13-21.pkl'
	load_model = utils.load_model(name)
	validate_proba = load_model.predict_proba(X_validate)[:, 1]
	validate_auc = roc_auc_score(y_validate, validate_proba)
	print validate_auc


def test_save_numpy():
	a = np.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
	np.savetxt(OUTPUT_DIR+"foo.csv", a, delimiter=",")


def func(df, haha):
	print '666', haha
	print df
	return df

if __name__ == '__main__':
	pass
	# df = pd.read_csv(INPUT_DIR+'order_products__prior_extend.csv')
	# print list(df.columns)
	# print df.head(2)
	#
	# df = pd.read_pickle(INPUT_DIR+'validate_diff_order_streaks_fix_add_new_core.pkl')
	# print df.shape
	# print df.dtypes
	# df.to_pickle(INPUT_DIR+'validate_1.pkl')
	# df = pd.read_csv(INPUT_DIR+'validate_diff_order_streaks_fix_add_new.csv_core.csv', compact_ints=True, index_col=0, dtype = {1:  np.float32})
	# print df.shape
	# print df.head().ix[:, [PID, 1, 2]]

	# print df[df['user_id'] == 202279][['user_id', 'product_id', 'user_product_bought_times', 'user_order_count']]
	# print df[df['user_id'] == 202279][['user_id', 'product_id', 'user_product_reorder_times', 'user_product_bought_times', 'user_product_reorder_ratio']]
	# print df[df['user_id'] == 202279][['user_id', 'product_id', 'user_active_days', 'user_bought_times/user_active_days', 'user_order_count/user_active_days']]

	# df = pd.read_csv(INPUT_DIR + 'validate_diff_order_streaks_fix_add_new.csv')
	# print df.shape
	# print df.dtypes
	# useful_feature = [
	# 	'last', 'prev1', 'prev2', 'median', 'mean',
	# 	'aisle_products',
	# 	'dep_products',
	# 	'prod_users_unq', 'prod_users_unq_reordered',
	# 	'up_order_rate_since_first_order',
	# 	'add_to_cart_order_inverted_mean', 'add_to_cart_order_relative_mean',
	# ]
	# print df.head()[useful_feature]

	# df = pd.DataFrame({'a': [1, 2, 3, np.nan]})
	# df.to_csv(INPUT_DIR+'my_test.csv', index=False)
	# df_2 = pd.read_csv(INPUT_DIR+'my_test.csv', compact_ints=True)
	# print df_2.dtypes
	# print df_2.a

	from sklearn import model_selection

	# groups = np.array([1, 1, 2, 2, 3, 3])
	# group_kfold = model_selection.GroupKFold(n_splits=2)
	# print type(group_kfold.split(groups, groups, groups))
	# X = None
	# y = None
	# for train_index, test_index in group_kfold.split(groups, groups, groups):
	# 	print train_index, test_index
		# print groups[train_index], groups[test_index]
	# x = tf.Variable(3, name='x')
	# y = tf.Variable(4, name='y')
	# f = x*x*y + y + 2
	# sess = tf.Session()

	n = 2
	for k in range(n + 1)[::-1]:
		print k

	print range(n + 1)

