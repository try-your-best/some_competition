# -*- coding: utf-8 -*-


from scipy import stats

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

	df_orders = df_orders.merge(df_reorder_size, how='left', on=[OID, UID])

	return df_orders


def get_recent_reorder_size(df, last_n):
	df = df.sort_values('order_number', ascending=False)
	last_size = df.iloc[0]['reorder_size']
	last_n_order = df.head(last_n)
	last_n_sizes = last_n_order['reorder_size']
	# stats.mode(last_n_sizes).mode[0]
	prefix = 'last_n_sizes_'
	return pd.Series([last_size, last_n_sizes.mean(), last_n_sizes.max(),
		last_n_sizes.min(), last_n_sizes.std()],
		index=['last_size', prefix+'mean', prefix+'max', prefix+'min', prefix+'std']
	)


def gen_recency_feature(df_prior_orders_extend):
	df_recency = df_prior_orders_extend.groupby([UID]).apply(get_recent_reorder_size, 3).reset_index()
	return df_recency


def get_user_reorder_size(df):
	reorder_sizes = df['reorder_size']
	prefix = 'reorder_sizes_'
	most = stats.mode(reorder_sizes)
	return pd.Series([reorder_sizes.mean(), reorder_sizes.std(),
		reorder_sizes.max(), reorder_sizes.min(), most.mode[0],  most.count[0]],
		index=[prefix+'mean', prefix+'std', prefix+'max', prefix+'min', prefix+'most', prefix+'most_count']
	)


def gen_entity_feature(df_prior_orders_extend):
	# prefix = 'reorder_sizes'
	# key = UID
	# df = df_prior_orders_extend.groupby(key).agg({
	# 	'reorder_size': {
	# 		'{}_mean'.format(prefix): np.mean,
	# 		'{}_std'.format(prefix): generate_feature.lambda_std,
	# 		'{}_max'.format(prefix): np.max,
	# 		'{}_min'.format(prefix): np.min,
	# 	},
	# }).reset_index()
	#
	# df.columns = [key] + list(df.columns.droplevel(0))[1:]
	df = df_prior_orders_extend.groupby(UID).apply(get_user_reorder_size).reset_index()
	return df


def gen_feature_by_rule(df):
	df['last_size-last_n_sizes_mean'] = df['last_size'] - df['last_n_sizes_mean']
	pass


@utils.timer
def generate_all_feature(df):
	df_prior_orders_extend = pd.read_csv(REORDER_INPUT_DIR+'prior_orders_extend.csv')
	df_target_prior_orders_extend = df_prior_orders_extend[df_prior_orders_extend[UID].isin(df[UID])]

	df_recency = gen_recency_feature(df_target_prior_orders_extend)
	# print df_recency.head()
	df = df.merge(df_recency, how='left', on=[UID])

	df_user_reorder_size = gen_entity_feature(df_target_prior_orders_extend)
	# print df_user_reorder_size.head()
	df = df.merge(df_user_reorder_size, how='left', on=[UID])

	return df


def add_all_feature(df, outfile):
	df = generate_all_feature(df)
	print df.head()
	df.to_csv(outfile, index=False)


# def add_user_reorder_size_feature(input_file, output_file):
# 	df = pd.read_csv(input_file)
# 	df_prior_orders_extend = pd.read_csv(REORDER_INPUT_DIR+'prior_orders_extend.csv')
# 	df_target_prior_orders_extend = df_prior_orders_extend[df_prior_orders_extend[UID].isin(df[UID])]
# 	df_user_reorder_size = gen_entity_feature(df_target_prior_orders_extend)
# 	print df_user_reorder_size.head()
#
# 	return df


def test_generate_feature():
	df_validate_tiny = sample(DataSetType.Validate)
	df_validate_tiny = generate_all_feature(df_validate_tiny)


if __name__ == '__main__':
	pass
	# df_validate = sample(DataSetType.Validate)
	# add_all_feature(df_validate, REORDER_INPUT_DIR+'validate_all_reorder.csv')

	# df_train = sample(DataSetType.Train)
	# add_all_feature(df_train, REORDER_INPUT_DIR+'train_all_reorder.csv')

	# generate_feature.create_feature_corr(REORDER_INPUT_DIR+'validate.csv', REORDER_INPUT_DIR+'corr.csv')
	# generate_feature.create_feature_corr(REORDER_INPUT_DIR+'train.csv', REORDER_INPUT_DIR+'corr_train.csv')
	generate_feature.create_feature_corr(REORDER_INPUT_DIR+'train_all_reorder.csv', REORDER_INPUT_DIR+'corr_all_reorder.csv')

	# test_generate_feature()

	# a = pd.Series([1, 1, 2])
	# print stats.mode(a).mode[0]
