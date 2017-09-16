# -*- coding: utf-8 -*-

from const import *
import utils
import generate_feature


@utils.timer
def sample(data_set_type):
	df_orders = pd.read_csv(ORDER_PATH)
	df_orders = df_orders[df_orders['eval_set'] == 'train'].drop(['eval_set'], axis=1)

	if data_set_type == DataSetType.Train:
		df_ground = generate_feature.load_ground_truth_train()
	else:
		df_ground = generate_feature.load_ground_truth_validate()

	df_orders = df_orders[df_orders[UID].isin(df_ground[UID])]

	df_reorder_size = pd.read_csv(REORDER_INPUT_DIR + 'train_true_reorder_size.csv')
	df_reorder_size['label'] = df_reorder_size['reorder_size'].map(lambda x: 1 if x == 0 else 0)
	df_reorder_size = df_reorder_size.drop(['reorder_size'], axis=1)

	df_orders = df_orders.merge(df_reorder_size, how='left', on=[OID, UID])

	return df_orders


def gen_feature_for_order():
	df_order_products__prior_extend = pd.read_csv(INPUT_DIR+'order_products__prior_extend.csv')
	key = [UID, OID]
	prefix = 'order'
	df = df_order_products__prior_extend.groupby(key).agg({
		OID: {'{}_size'.format(prefix): np.size},
		'reordered': {'reorder_ratio': np.mean}
	}).reset_index()

	df.columns = key + list(df.columns.droplevel(0))[2:]

	df_order_extend = pd.read_csv(REORDER_INPUT_DIR+'prior_orders_extend.csv')
	df_order_extend = df_order_extend.merge(df, how='left', on=key)
	df_order_extend.to_csv(NONE_ORDER_INPUT_DIR+'prior_orders_extend.csv', index=False)


def get_recency_feature(df, last_n):
	df = df.sort_values('order_number', ascending=False)
	last_reorder_size = df.iloc[0]['reorder_size']
	last_reorder_ratio = df.iloc[0]['reorder_ratio']
	last_order_none = 1 if last_reorder_size == 0 else 0

	prefix = 'last_{}'.format(last_n)
	df_last_n_order = df.head(last_n)
	last_n_order = get_order_feature(df_last_n_order, prefix+'_none_order', prefix+'_order', prefix+'_reorder')
	return pd.Series([last_reorder_size, last_order_none, last_reorder_ratio],
		index=['last_reorder_size', 'last_order_none', 'last_reorder_ratio']).append(last_n_order)


def get_order_feature(df_order, none_prefix, prior_prefix, reorder_prefix):
	df_none_order = df_order[df_order['reorder_size'] == 0]

	none_order_count = df_none_order.shape[0]
	none_order_ratio = none_order_count / float(df_order.shape[0])

	# if df_none_order.empty:
	# 	none_order_dow_mean = np.nan
	# 	none_order_dow_std = np.nan
	# 	none_order_hour_of_day_mean = np.nan
	# 	none_order_hour_of_day_std = np.nan
	# else:
	# order_dow_mean = df_none_order['order_dow'].mean()
	# order_dow_std = df_none_order['order_dow'].std()
	# order_hour_of_day_mean = df_none_order['order_hour_of_day'].mean()
	# order_hour_of_day_std = df_none_order['order_hour_of_day'].std()

	order_dow = df_order['order_dow']
	order_hour_of_day = df_order['order_hour_of_day']
	reorder_size = df_order['reorder_size']
	days_since_prior_order = df_order['days_since_prior_order']
	reorder_ratio = df_order['reorder_ratio']

	return pd.Series([none_order_count, none_order_ratio,
		order_dow.mean(), order_dow.std(),
		order_hour_of_day.mean(), order_hour_of_day.std(),
		days_since_prior_order.mean(), days_since_prior_order.std(),
		days_since_prior_order.max(), days_since_prior_order.min(),
		reorder_size.mean(), reorder_size.std(), reorder_size.max(), reorder_size.min(),
		reorder_ratio.mean(), reorder_ratio.std(), reorder_ratio.max(), reorder_ratio.min(),
		],
		index=[
			none_prefix + '_count', none_prefix + '_ratio',
			prior_prefix + '_dow_mean', prior_prefix + '_dow_std',
			prior_prefix + '_hour_of_day_mean', prior_prefix + '_hour_of_day_std',
			prior_prefix + '_days_since_prior_order_mean', prior_prefix + '_days_since_prior_order_std',
			prior_prefix + '_days_since_prior_order_max', prior_prefix + '_days_since_prior_order_min',
			reorder_prefix + '_size_mean', reorder_prefix + '_size_std',
			reorder_prefix + '_size_max', reorder_prefix + '_size_min',
			reorder_prefix + '_ratio_mean', reorder_prefix + '_ratio_std',
			reorder_prefix + '_ratio_max', reorder_prefix + '_ratio_min',
		]
	)


def get_order_feature_group(df_order):
	return get_order_feature(df_order, 'none_order', 'prior_order', 'reorder')


def gen_recency_feature(df_prior_orders_extend):
	df_recency = df_prior_orders_extend.groupby([UID]).apply(get_recency_feature, 3).reset_index()
	return df_recency


def gen_user_feature(df_prior_orders_extend):
	return df_prior_orders_extend.groupby(UID).apply(get_order_feature_group).reset_index()


def gen_feature_by_rule(df):
	df['days_since_prior_order_30'] = 0
	df.ix[df['days_since_prior_order'] == 30, 'days_since_prior_order_30'] = 1
	return df


@utils.timer
def gen_basic_feature(df):
	df_prior_orders_extend = pd.read_csv(NONE_ORDER_INPUT_DIR+'prior_orders_extend.csv')
	df_target_prior_orders_extend = df_prior_orders_extend[df_prior_orders_extend[UID].isin(df[UID])]
	df_recency = gen_recency_feature(df_target_prior_orders_extend)
	# print df_recency.head()
	df = df.merge(df_recency, how='left', on=[UID])

	df_user = gen_user_feature(df_target_prior_orders_extend)
	# print df_user.head()
	df = df.merge(df_user, how='left', on=[UID])

	df = gen_feature_by_rule(df)
	print df.head(3)

	return df


def add_basic_feature(df, output_file):
	df = gen_basic_feature(df)
	df.to_csv(output_file, index=False)


if __name__ == '__main__':
	pass
	# df_validate = sample(DataSetType.Validate)
	# add_basic_feature(df_validate, NONE_ORDER_INPUT_DIR+'validate_ratio.csv')

	# df = pd.read_csv(NONE_ORDER_INPUT_DIR+'validate_ratio.csv')
	# df = df.rename(columns={'last_size': 'last_reorder_size'})
	# df.to_csv(NONE_ORDER_INPUT_DIR+'validate_ratio.csv', index=False)

	df_train = sample(DataSetType.Train)
	add_basic_feature(df_train, NONE_ORDER_INPUT_DIR + 'train_ratio.csv')
	#
	generate_feature.create_feature_corr(NONE_ORDER_INPUT_DIR + 'train_ratio.csv',
		NONE_ORDER_INPUT_DIR + 'corr_ratio.csv')

	# gen_feature_for_order()

