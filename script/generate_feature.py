# -*- coding: utf-8 -*-

import gc

import pandas as pd
import numpy as np

from const import *
from utils import timer


# GROUND_TRUTH_TRAIN_PATH = '../input/ground_truth_train_sample.csv'
# GROUND_TRUTH_VALIDATE_PATH = '../input/ground_truth_validate_sample.csv'

GROUND_TRUTH_TRAIN_PATH = '../input/ground_truth_train.csv'
GROUND_TRUTH_VALIDATE_PATH = '../input/ground_truth_validate.csv'


def load_prior_order_products():
	df = pd.read_csv(ORDER_PRODUCT_PATH, compact_ints=True)
	# df = df[df[UID] <= 20]
	return df


def load_ground_truth_train():
	df = pd.read_csv(GROUND_TRUTH_TRAIN_PATH)
	# df = df[df[UID] <= 20]
	return df


def load_ground_truth_validate():
	df = pd.read_csv(GROUND_TRUTH_VALIDATE_PATH)

	# test
	# df = df.head(1000)
	return df


@timer
def sample(data_set_type):
	print 'sample data_set_type :{}'.format(data_set_type)
	df_orders = pd.read_csv(ORDER_PATH)
	df_orders = df_orders[df_orders['eval_set'] == 'train'].drop([OID, 'eval_set'], axis=1)

	if data_set_type == DataSetType.Train:
		df_ground = load_ground_truth_train()
	else:
		df_ground = load_ground_truth_validate()

	df_ground['label'] = 1

	df_orders = df_orders[df_orders[UID].isin(df_ground[UID])]

	df_order_products = pd.read_csv(ORDER_PRODUCT_PATH)

	df_order_products = df_order_products[df_order_products[UID].isin(
		df_orders[UID])][[UID, PID, AID, DID]].drop_duplicates()

	# add order context
	df_order_products = df_order_products.merge(df_orders, how='left', on=[UID])

	df_order_products = df_order_products.merge(df_ground, how='left', on=[UID, PID])

	df_order_products.label = df_order_products.label.fillna(0)
	# df_order_products[['label']] = df_order_products[['days_since_prior_order', 'label']].fillna(0)

	return df_order_products


def sample_test_set():
	df_orders = pd.read_csv(ORDER_PATH)
	df_orders = df_orders[df_orders['eval_set'] == 'test'].drop([OID, 'eval_set'], axis=1)

	df_order_products = pd.read_csv(ORDER_PRODUCT_PATH)

	df_order_products = df_order_products[df_order_products[UID].isin(
		df_orders[UID])][[UID, PID, AID, DID]].drop_duplicates()

	df_order_products = df_order_products.merge(df_orders, how='left', on=[UID])

	return df_order_products


@timer
def gen_user_interaction_feature(key, df_order_products):
	print 'gen_user_interaction_feature, key: {}'.format(key)

	prefix = {PID: 'user_product', AID: 'user_aisle', DID: 'user_department'}[key]

	key = [UID, key]

	df = df_order_products.groupby(key).agg({
		OID: {'{}_bought_times'.format(prefix): np.size},
		'reordered': {
			'{}_reorder_times'.format(prefix): np.sum,
			'{}_reorder_ratio'.format(prefix): np.mean,
		},
		'add_to_cart_order': {
			'{}_add_to_cart_order_max'.format(prefix): np.max,
			'{}_add_to_cart_order_min'.format(prefix): np.min,
			'{}_add_to_cart_order_mean'.format(prefix): np.mean,
		},
		'order_number': {
			'{}_order_number_max'.format(prefix): np.max,
			'{}_order_number_min'.format(prefix): np.min,
		},
		'order_dow': {'{}_order_dow_mean'.format(prefix): np.mean},
		'order_hour_of_day': {'{}_order_hour_of_day_mean'.format(prefix): np.mean},
		'days_since_prior_order': {
			'{}_days_since_prior_order_max'.format(prefix): np.max,
			'{}_days_since_prior_order_min'.format(prefix): np.min,
			'{}_days_since_prior_order_mean'.format(prefix): np.mean,
		},
	}).reset_index()

	df.columns = key + list(df.columns.droplevel(0))[2:]

	return df


def lambda_std(x):
	# 直接 std 与 用函数调用的行为不一致
	return np.std(x)


@timer
def gen_user_interaction_feature_std(key, df_order_products):
	print 'gen_user_interaction_feature_std, key: {}'.format(key)

	prefix = {PID: 'user_product', AID: 'user_aisle', DID: 'user_department'}[key]

	key = [UID, key]

	df = df_order_products.groupby(key).agg({
		'add_to_cart_order': {
			'{}_add_to_cart_order_std'.format(prefix): lambda_std,
		},
		'order_dow': {'{}_order_dow_std'.format(prefix): lambda_std},
		'order_hour_of_day': {'{}_order_hour_of_day_std'.format(prefix): lambda_std},
		'days_since_prior_order': {
			'{}_days_since_prior_order_std'.format(prefix): lambda_std,
		},
	}).reset_index()

	df.columns = key + list(df.columns.droplevel(0))[2:]

	return df


def gen_recent_user_interaction_feature(key, df_order_products):
	# 用户订单数是4-100, 最近的3个订单反应情况

	prefix = {PID: 'user_product', AID: 'user_aisle', DID: 'user_department'}[key]

	prefix = prefix + 'recent'

	key = [UID, key]

	df = df_order_products.groupby(key).agg({
		OID: {'{}_bought_times'.format(prefix): np.size},
		'reordered': {
			'{}_reorder_times'.format(prefix): np.sum,
			'{}_reorder_ratio'.format(prefix): np.mean,
		},
		'add_to_cart_order': {
			'{}_add_to_cart_order_max'.format(prefix): np.max,
			'{}_add_to_cart_order_min'.format(prefix): np.min,
			'{}_add_to_cart_order_mean'.format(prefix): np.mean,
		},
		'order_number': {
			'{}_order_number_max'.format(prefix): np.max,
			'{}_order_number_min'.format(prefix): np.min,
		},
		'order_dow': {'{}_order_dow_mean'.format(prefix): np.mean},
		'order_hour_of_day': {'{}_order_hour_of_day_mean'.format(prefix): np.mean},
		'days_since_prior_order': {
			'{}_days_since_prior_order_max'.format(prefix): np.max,
			'{}_days_since_prior_order_min'.format(prefix): np.min,
			'{}_days_since_prior_order_mean'.format(prefix): np.mean,
		},
	}).reset_index()

	df.columns = key + list(df.columns.droplevel(0))[2:]

	return df


def gen_recent_feature():
	pass


@timer
def gen_entity_feature(key, df_order_products):
	print 'gen_entity_feature, key: {}'.format(key)

	prefix = {UID: 'user', PID: 'product', AID: 'aisle', DID: 'department'}[key]

	df = df_order_products.groupby(key).agg({
		UID: {'{}_bought_times'.format(prefix): np.size},
		OID: {'{}_order_count'.format(prefix): pd.Series.nunique},
		PID: {'{}_product_count'.format(prefix): pd.Series.nunique},
		'reordered': {
			'{}_reorder_times'.format(prefix): np.sum,
			'{}_reorder_ratio'.format(prefix): np.mean,
		},
		'add_to_cart_order': {'{}_add_to_cart_order_mean'.format(prefix): np.mean},
		'order_dow': {'{}_order_dow_mean'.format(prefix): np.mean},
		'order_hour_of_day': {'{}_order_hour_of_day_mean'.format(prefix): np.mean},
		'days_since_prior_order': {
			'{}_days_since_prior_order_mean'.format(prefix): np.mean,
		# 	'{}_active_days'.format(prefix): np.sum, # 单个订单里的多个产品重复计算没意义
		},
		'order_number': {
			'{}_order_number_max'.format(prefix): np.max,
			# '{}_order_number_min'.format(prefix): np.min,
		},
	}).reset_index()

	df.columns = [key] + list(df.columns.droplevel(0))[1:]
	return df


@timer
def gen_entity_feature_std(key, df_order_products):
	# TODO 这个 std 感觉没有意义
	print 'gen_entity_feature_std, key: {}'.format(key)

	prefix = {UID: 'user', PID: 'product', AID: 'aisle', DID: 'department'}[key]

	df = df_order_products.groupby(key).agg({
		'add_to_cart_order': {'{}_add_to_cart_order_std'.format(prefix): lambda_std},
		'order_dow': {'{}_order_dow_std'.format(prefix): lambda_std},
		'order_hour_of_day': {'{}_order_hour_of_day_std'.format(prefix): lambda_std},
		'days_since_prior_order': {
			'{}_days_since_prior_order_std'.format(prefix): lambda_std,
		},
	}).reset_index()

	df.columns = [key] + list(df.columns.droplevel(0))[1:]
	return df


@timer
def gen_ratio_feature(df):

	df = gen_user_interaction_ratio(df, UP, UA)
	df = gen_user_interaction_ratio(df, UP, UD)

	df = gen_entity_ratio(df, 'product', 'aisle')
	df = gen_entity_ratio(df, 'product', 'department')

	df = gen_ratio_by_rule(df)

	return df


def gen_user_interaction_ratio(df, prefix1, prefix2):
	for action in [
		'bought_times',
		'reorder_times', 'reorder_ratio',
		'add_to_cart_order_min', 'add_to_cart_order_mean',
		'order_number_max', 'order_number_min']:
		ratio_col = '_'.join([prefix1+'/'+prefix2, action])
		col1 = '_'.join([prefix1, action])
		col2 = '_'.join([prefix2, action])
		df[ratio_col] = df[col1] / df[col2]

	return df


def gen_entity_ratio(df, prefix1, prefix2):
	for action in [
		'bought_times',
		'reorder_times', 'reorder_ratio',
		'add_to_cart_order_mean']:
		ratio_col = '_'.join([prefix1+'/'+prefix2, action])
		col1 = '_'.join([prefix1, action])
		col2 = '_'.join([prefix2, action])
		df[ratio_col] = df[col1] / df[col2]

	return df


def gen_ratio_by_rule(df):
	df['user_product_bought_times/user_bought_times'] = df['user_product_bought_times'] / df['user_bought_times']
	df['user_product_bought_times/user_order_count'] = df['user_product_bought_times'] / df['user_order_count']
	df['user_aisle_bought_times/user_bought_times'] = df['user_aisle_bought_times'] / df['user_bought_times']
	df['user_aisle_bought_times/user_order_count'] = df['user_aisle_bought_times'] / df['user_order_count']
	df['user_department_bought_times/user_bought_times'] = df['user_department_bought_times'] / df['user_bought_times']
	df['user_department_bought_times/user_order_count'] = df['user_department_bought_times'] / df['user_order_count']

	df['user_product_order_number_max/user_order_number_max'] = \
		df['user_product_order_number_max'] / df['user_order_number_max']
	df['user_aisle_order_number_max/user_order_number_max'] = \
		df['user_aisle_order_number_max'] / df['user_order_number_max']
	df['user_department_order_number_max/user_order_number_max'] = \
		df['user_department_order_number_max'] / df['user_order_number_max']

	df['user_order_size_mean'] = df['user_bought_times'] / df['user_order_count']

	df['user_product_add_to_cart_order_mean/user_order_size_mean'] = \
		df['user_product_add_to_cart_order_mean'] / df['user_order_size_mean']
	df['user_product_add_to_cart_order_min/user_order_size_mean'] = \
		df['user_product_add_to_cart_order_min'] / df['user_order_size_mean']
	df['user_aisle_add_to_cart_order_mean/user_order_size_mean'] = \
		df['user_aisle_add_to_cart_order_mean'] / df['user_order_size_mean']
	df['user_aisle_add_to_cart_order_min/user_order_size_mean'] = \
		df['user_aisle_add_to_cart_order_min'] / df['user_order_size_mean']
	df['user_department_add_to_cart_order_mean/user_order_size_mean'] = \
		df['user_department_add_to_cart_order_mean'] / df['user_order_size_mean']
	df['user_department_add_to_cart_order_min/user_order_size_mean'] = \
		df['user_department_add_to_cart_order_min'] / df['user_order_size_mean']

	df['user_product_reorder_ratio/user_reorder_ratio'] = df['user_product_reorder_ratio'] / df['user_reorder_ratio']
	df['user_aisle_reorder_ratio/user_reorder_ratio'] = df['user_aisle_reorder_ratio'] / df['user_reorder_ratio']
	df['user_department_reorder_ratio/user_reorder_ratio'] = \
		df['user_department_reorder_ratio'] / df['user_reorder_ratio']

	df['user_product_bought_times/user_product_order_number_max-user_product_order_number_min'] = \
		df['user_product_bought_times'] / (df['user_product_order_number_max'] + 1 - df['user_product_order_number_min'])

	df['user_product_days_since_prior_order/user_product_days_since_prior_order_max'] = \
		df['days_since_prior_order'] / df['user_product_days_since_prior_order_max']
	df['user_product_days_since_prior_order/user_product_days_since_prior_order_min'] = \
		df['days_since_prior_order'] / df['user_product_days_since_prior_order_min']
	df['user_product_days_since_prior_order/user_product_days_since_prior_order_mean'] = \
		df['days_since_prior_order'] / df['user_product_days_since_prior_order_mean']

	df['user_bought_times/user_active_days'] = df['user_bought_times'] / df['user_active_days']
	df['user_order_count/user_active_days'] = df['user_order_count'] / df['user_active_days']

	return df


def gen_diff_by_rule(df):
	df['order_number-user_product_order_number_max'] = df['order_number'] - df['user_product_order_number_max']
	df['order_number-user_aisle_order_number_max'] = df['order_number'] - df['user_aisle_order_number_max']
	df['order_number-user_department_order_number_max'] = df['order_number'] - df['user_department_order_number_max']

	return df


def gen_rank_feature(df):
	df = gen_user_interaction_rank(df, UP, UA, [UID, AID])
	df = gen_user_interaction_rank(df, UP, UD, [UID, DID])

	df = gen_entity_rank(df, 'product', 'aisle', [AID])
	df = gen_entity_rank(df, 'product', 'department', [DID])

	df = gen_rank_by_rule(df)

	return df


@timer
def gen_user_interaction_rank(df, prefix1, prefix2, key):
	print 'gen_user_interaction_rank', prefix1, prefix2, key
	for action in [
		'bought_times',
		'reorder_times', 'reorder_ratio',
		'add_to_cart_order_min', 'add_to_cart_order_mean',
		'order_number_max', 'order_number_min',
	]:
		col1 = '_'.join([prefix1+'/'+prefix2, action])

		col2 = '_'.join([col1, 'dense_rank'])
		col3 = '_'.join([col1, 'pct_rank'])
		col4 = '_'.join([col1, 'rank'])

		cols = list(key)
		cols.append(col1)

		df[col2] = df[cols].groupby(key).rank(method='dense', ascendinsg=False)
		df[col3] = df[cols].groupby(key).rank(pct=True, ascending=False)
		df[col4] = df[col2] * df[col3]

	return df


@timer
def gen_entity_rank(df, prefix1, prefix2, key):
	print 'gen_entity_rank', prefix1, prefix2, key

	for action in [
		'bought_times',
		'reorder_times', 'reorder_ratio',
		'add_to_cart_order_mean',
	]:
		col1 = '_'.join([prefix1+'/'+prefix2, action])

		col2 = '_'.join([col1, 'dense_rank'])
		col3 = '_'.join([col1, 'pct_rank'])
		col4 = '_'.join([col1, 'rank'])

		cols = list(key)
		cols.append(col1)

		df[col2] = df[cols].groupby(key).rank(method='dense', ascending=False)
		df[col3] = df[cols].groupby(key).rank(pct=True, ascending=False)
		df[col4] = df[col2] * df[col3]

	return df


@timer
def gen_rank_by_rule(df):
	key = [UID]
	for action in [
		'user_product_bought_times/user_bought_times',
		'user_product_bought_times/user_order_count',
		'user_aisle_bought_times/user_bought_times',
		'user_aisle_bought_times/user_bought_times',
		'user_aisle_bought_times/user_order_count',
		'user_department_bought_times/user_bought_times',
		'user_department_bought_times/user_order_count',
		'user_product_order_number_max/user_order_number_max',
		'user_aisle_order_number_max/user_order_number_max',
		'user_department_order_number_max/user_order_number_max',
		'user_product_add_to_cart_order_mean/user_order_size_mean',
		'user_product_add_to_cart_order_min/user_order_size_mean',
		'user_aisle_add_to_cart_order_mean/user_order_size_mean',
		'user_aisle_add_to_cart_order_min/user_order_size_mean',
		'user_department_add_to_cart_order_mean/user_order_size_mean',
		'user_department_add_to_cart_order_min/user_order_size_mean',
		'user_product_reorder_ratio/user_reorder_ratio',
		'user_aisle_reorder_ratio/user_reorder_ratio',
		'user_department_reorder_ratio/user_reorder_ratio',
		'user_product_bought_times/user_product_order_number_max-user_product_order_number_min',
		'user_product_days_since_prior_order/user_product_days_since_prior_order_max',
		'user_product_days_since_prior_order/user_product_days_since_prior_order_min',
		'user_product_days_since_prior_order/user_product_days_since_prior_order_mean',
		'user_bought_times/user_active_days',
		'user_order_count/user_active_days',
	]:
		col1 = action + '_dense_rank'
		col2 = action + '_pct_rank'
		col3 = action + '_rank'
		cols = list(key)
		cols.append(action)

		df[col1] = df[cols].groupby(key).rank(method='dense', ascending=False)
		df[col2] = df[cols].groupby(key).rank(pct=True, ascending=False)
		df[col3] = df[col1] * df[col2]

	return df


def gen_std_feature(df):
	df_prior_order_products = load_prior_order_products()
	df_target_user_order_products = df_prior_order_products[df_prior_order_products[UID].isin(df[UID])]

	# df_user_product = gen_user_interaction_feature_std(PID, df_target_user_order_products)
	# print gc.collect()
	# df = df.merge(df_user_product, how='left', on=[UID, PID], copy=False)
	# print gc.collect()
	#
	# df_user_aisle = gen_user_interaction_feature_std(AID, df_target_user_order_products)
	# print gc.collect()
	# df = df.merge(df_user_aisle, how='left', on=[UID, AID], copy=False)
	# print gc.collect()

	df_user_department = gen_user_interaction_feature_std(DID, df_target_user_order_products)
	print gc.collect()
	df = df.merge(df_user_department, how='left', on=[UID, DID], copy=False)
	print gc.collect()

	df_user = gen_entity_feature_std(UID, df_target_user_order_products)
	print gc.collect()
	df = df.merge(df_user, how='left', on=[UID], copy=False)
	print gc.collect()

	del df_target_user_order_products
	gc.collect()

	df_product = gen_entity_feature_std(PID, df_prior_order_products)
	print gc.collect()
	df = df.merge(df_product, how='left', on=[PID], copy=False)
	print gc.collect()

	df_aisle = gen_entity_feature_std(AID, df_prior_order_products)
	print gc.collect()
	df = df.merge(df_aisle, how='left', on=[AID], copy=False)
	print gc.collect()

	df_department = gen_entity_feature_std(DID, df_prior_order_products)
	print gc.collect()
	df = df.merge(df_department, how='left', on=[DID], copy=False)

	return df


def generate_feature(df):
	df_prior_order_products = load_prior_order_products()

	df_target_user_order_products = df_prior_order_products[df_prior_order_products[UID].isin(df[UID])]

	df_user_product = gen_user_interaction_feature(PID, df_target_user_order_products)
	df = df.merge(df_user_product, how='left', on=[UID, PID])

	df_user_aisle = gen_user_interaction_feature(AID, df_target_user_order_products)
	df = df.merge(df_user_aisle, how='left', on=[UID, AID])

	df_user_department = gen_user_interaction_feature(DID, df_target_user_order_products)
	df = df.merge(df_user_department, how='left', on=[UID, DID])

	df_user = gen_entity_feature(UID, df_target_user_order_products)
	df = df.merge(df_user, how='left', on=[UID])

	df_product = gen_entity_feature(PID, df_prior_order_products)
	df = df.merge(df_product, how='left', on=[PID])

	df_aisle = gen_entity_feature(AID, df_prior_order_products)
	df = df.merge(df_aisle, how='left', on=[AID])

	df_department = gen_entity_feature(DID, df_prior_order_products)
	df = df.merge(df_department, how='left', on=[DID])

	feature_num1 = df.columns.size
	print 'basic feature', feature_num1
	#
	# df = gen_ratio_feature(df)
	# feature_num2 = df.columns.size
	# print 'ration feature', feature_num2 - feature_num1
	#
	# df = gen_rank_feature(df)
	# feature_num3 = df.columns.size
	# print 'rank feature', feature_num3 - feature_num2
	#
	# df = gen_diff_by_rule(df)
	# print 'total feature', df.columns.size

	return df


@timer
def create_feature_corr(input_file, output_file):
	df = pd.read_csv(input_file)
	df.corr().to_csv(output_file)


@timer
def add_ratio_feature(input_file, output_file):
	df = pd.read_csv(input_file)
	df = gen_ratio_feature(df)
	df.to_csv(output_file, index=False)


@timer
def add_std_feature(input_file, output_file):
	df = pd.read_csv(input_file, compact_ints=True)
	df = gen_std_feature(df)
	print gc.collect()
	df.to_csv(output_file, index=False)


@timer
def add_rank_diff_feature(input_file, output_file):
	df = pd.read_csv(input_file)
	df = gen_rank_feature(df)
	df = gen_diff_by_rule(df)
	df.to_csv(output_file, index=False)


@timer
def add_rank_by_step(input_file, output_file):
	df = pd.read_csv(input_file, compact_ints=True)
	df = gen_user_interaction_rank(df, UP, UA, [UID, AID])
	print 'step1', gc.collect()

	df = gen_user_interaction_rank(df, UP, UD, [UID, DID])
	print 'step2', gc.collect()

	# df = gen_entity_rank(df, 'product', 'aisle', [AID])
	# print gc.collect()
	# df = gen_entity_rank(df, 'product', 'department', [DID])
	# print gc.collect()
	#
	# df = gen_rank_by_rule(df)

	df.to_csv(output_file, index=False)


@timer
def add_diff_feature(input_file, output_file):
	df = pd.read_csv(input_file, compact_ints=True)
	df = gen_diff_by_rule(df)
	df.to_csv(output_file, index=False)


def add_order_streaks(input_file, output_file):
	df = pd.read_csv(input_file)
	df_order_streaks = pd.read_csv(INPUT_DIR+'order_streaks.csv')
	df = df.merge(df_order_streaks, how='left', on=[UID, PID])
	df.to_csv(output_file, index=False)


def some_test():
	pass
	# add_ratio_feature(INPUT_DIR + 'validate_sample.csv', INPUT_DIR + 'validate_sample_ratio.csv')
	# add_rank_by_step(INPUT_DIR+'validate_sample_ratio.csv', INPUT_DIR+'validate_sample_interaction_rank.csv')
	# add_diff_feature(INPUT_DIR + 'validate_sample_ratio.csv', INPUT_DIR+'validate_sample_diff.csv')
	add_std_feature(INPUT_DIR+'validate_sample_diff.csv', INPUT_DIR+'validate_sample_std.csv')


def fix_feature(source_feature_file, des_feature_file):
	df_orders = pd.read_csv(ORDER_PATH)
	df_orders = df_orders[df_orders['eval_set'] == 'prior']

	key = UID

	df_user_stat = df_orders.groupby(key).agg(
		{'days_since_prior_order': {
			'user_active_days': np.sum,
			'user_order_days_since_prior_mean': np.mean,
			'user_order_days_since_prior_median': np.median,
		}}
	).reset_index()

	df_user_stat.columns = [key] + list(df_user_stat.columns.droplevel(0))[1:]

	print df_user_stat.head()

	df_source = pd.read_csv(source_feature_file)
	df_source.drop(['user_active_days', 'department_active_days', 'product_active_days', 'aisle_active_days'], axis=1, inplace=True)
	df_source = df_source.merge(df_user_stat, how='left', on=[UID])
	# gc.collect()
	df_source['user_bought_times/user_active_days'] = df_source['user_bought_times'] / df_source['user_active_days']
	df_source['user_order_count/user_active_days'] = df_source['user_order_count'] / df_source['user_active_days']
	df_source.to_csv(des_feature_file, index=False)


def fix_feature_2(source_feature_file, des_feature_file):
	df_source = pd.read_csv(source_feature_file)
	# print 'before', df_source['user_product_bought_times/user_product_order_number_max-user_product_order_number_min'].isnull().sum()
	df_source['user_product_bought_times/user_product_order_number_max-user_product_order_number_min'] = \
		df_source['user_product_bought_times'] / (df_source['user_product_order_number_max'] + 1 - df_source['user_product_order_number_min'])
	# print 'after', df_source['user_product_bought_times/user_product_order_number_max-user_product_order_number_min'].isnull().sum()
	df_source.to_csv(des_feature_file, index=False)


def add_new_feature(source_file, new_feature_file, output_file):
	df_new_feature = pd.read_pickle(new_feature_file)
	df_source = pd.read_csv(source_file)
	df_source = df_source.merge(df_new_feature, how='left', on=[UID, PID])
	df_source.to_csv(output_file, index=False)


def drop_feature():
	no_use_cols = ['product_product_count', 'aisle_order_number_max', 'department_order_number_max']
	pass


if __name__ == '__main__':
	pass

	# df_train = sample(DataSetType.Train)
	# df_train = generate_feature(df_train)
	# df_train.to_csv('../input/train.csv', index=False)

	# df_validate = sample(DataSetType.Validate)
	# df_validate = generate_feature(df_validate)
	# df_validate.to_csv('../input/validate.csv', index=False)

	# df_test = sample_test_set()
	# df_test = generate_feature(df_test)
	# df_test.to_csv('../input/test.csv', index=False)

	# add_ratio_feature('../input/train.csv', '../input/train_ratio.csv')
	# add_ratio_feature('../input/validate.csv', '../input/validate_ratio.csv')
	# add_ratio_feature(INPUT_DIR+'test.csv', INPUT_DIR+'test_ratio.csv')

	# add_rank_diff_feature(INPUT_DIR+'train_ratio.csv', INPUT_DIR+'train_rank_diff.csv')
	# create_feature_corr(INPUT_DIR+'validate_ratio.csv', INPUT_DIR+'corr_ratio.csv')

	# add_rank_by_step(INPUT_DIR + 'train_ratio.csv', INPUT_DIR + 'train_interaction_rank.csv')
	# add_rank_by_step(INPUT_DIR + 'validate_ratio.csv', INPUT_DIR + 'validate_interaction_rank.csv')
	# add_rank_by_step(INPUT_DIR + 'validate_ratio.csv', INPUT_DIR + 'validate_interaction_rank.csv')
	# create_feature_corr(INPUT_DIR + 'validate_interaction_rank.csv', INPUT_DIR + 'corr_interaction_rank.csv')

	# add_diff_feature(INPUT_DIR+'train_ratio.csv', INPUT_DIR + 'train_diff.csv')
	# gc.collect()
	# add_diff_feature(INPUT_DIR + 'validate_ratio.csv', INPUT_DIR + 'validate_diff.csv')
	# add_diff_feature(INPUT_DIR + 'test_ratio.csv', INPUT_DIR + 'test_diff.csv')
	# create_feature_corr(INPUT_DIR + 'validate_diff.csv', INPUT_DIR + 'corr_diff.csv')

	# add_std_feature(INPUT_DIR+'validate_diff.csv', INPUT_DIR+'validate_std.csv')
	# add_std_feature(INPUT_DIR+'train_diff.csv', INPUT_DIR+'train_std_1.csv')
	# add_std_feature(INPUT_DIR+'train_std_1.csv', INPUT_DIR+'train_std.csv')
	# create_feature_corr(INPUT_DIR + 'validate_std.csv', INPUT_DIR + 'corr_std.csv')

	# add_order_streaks(INPUT_DIR+'validate_std.csv', INPUT_DIR+'validate_order_streaks.csv')
	# add_order_streaks(INPUT_DIR+'train_std.csv', INPUT_DIR+'train_order_streaks.csv')
	# some_test()

	# add_order_streaks(INPUT_DIR + 'validate_diff.csv', INPUT_DIR + 'validate_diff_order_streaks.csv')
	# add_order_streaks(INPUT_DIR + 'train_diff.csv', INPUT_DIR + 'train_diff_order_streaks.csv')
	# add_order_streaks(INPUT_DIR + 'test_diff.csv', INPUT_DIR + 'test_diff_order_streaks.csv')

	# fix_feature(INPUT_DIR + 'validate_diff_order_streaks.csv', INPUT_DIR + 'validate_diff_order_streaks_fix.csv')
	# fix_feature(INPUT_DIR + 'train_diff_order_streaks.csv', INPUT_DIR + 'train_diff_order_streaks_fix.csv')
	# fix_feature(INPUT_DIR + 'test_diff_order_streaks.csv', INPUT_DIR + 'test_diff_order_streaks_fix.csv')

	# add_new_feature(INPUT_DIR+'validate_diff_order_streaks_fix.csv', NEW_FEATURE_DIR+'train_submit_fix_2_output.pkl',
	# 	INPUT_DIR+'validate_diff_order_streaks_fix_add_new.csv')
	# add_new_feature(INPUT_DIR + 'train_diff_order_streaks_fix.csv', NEW_FEATURE_DIR + 'train_submit_fix_2_output.pkl',
	# 	INPUT_DIR + 'train_diff_order_streaks_fix_add_new.csv')
	# add_new_feature(INPUT_DIR + 'test_diff_order_streaks_fix.csv', NEW_FEATURE_DIR + 'test_submit_fix_2_output.pkl',
	# 	INPUT_DIR + 'test_diff_order_streaks_fix_add_new.csv')

	# create_feature_corr(INPUT_DIR + 'train_diff_order_streaks_fix_add_new.csv', INPUT_DIR+'corr_diff_order_streaks_fix_add_new.csv')

	# fix_feature_2(INPUT_DIR+'validate_diff_order_streaks_fix_add_new.csv', INPUT_DIR+'validate_diff_order_streaks_fix2_add_new.csv')
	# fix_feature_2(INPUT_DIR+'train_diff_order_streaks_fix_add_new.csv', INPUT_DIR+'train_diff_order_streaks_fix2_add_new.csv')
	fix_feature_2(INPUT_DIR+'test_diff_order_streaks_fix_add_new.csv', INPUT_DIR+'test_diff_order_streaks_fix2_add_new.csv')




