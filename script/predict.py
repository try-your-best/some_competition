# -*- coding: utf-8 -*-

import gc


import pandas as pd

import utils
from const import *
from ensemble import EnsembleClassifier
from sklearn import metrics

@utils.timer
def predict(models, input_file, output_file):
	df = pd.read_csv(input_file)

	df_result = df[[UID, PID]]
	df = utils.preprocess_reorder_product_data(df)

	X = df.drop(['label'], axis=1) if 'label' in df.columns else df

	del df
	gc.collect()

	dfs = []

	for model in models:
		proba = model.predict_proba(X)
		df = df_result.copy()
		df['proba'] = proba[:, 1]
		dfs.append(df)

	df = utils.concat(dfs)
	df = df.groupby([UID, PID]).mean().reset_index()
	df = df.sort_values('proba', ascending=False).reset_index(drop=True)
	df.to_csv(output_file, index=False)


train_set_features = ['product_id', 'aisle_id', 'department_id', 'order_number', 'order_dow', 'order_hour_of_day',
	'days_since_prior_order', 'user_product_add_to_cart_order_mean', 'user_product_add_to_cart_order_min',
	'user_product_bought_times', 'user_product_days_since_prior_order_max', 'user_product_days_since_prior_order_min',
	'user_product_days_since_prior_order_mean', 'user_product_order_hour_of_day_mean', 'user_product_order_dow_mean',
	'user_product_order_number_max', 'user_product_order_number_min', 'user_aisle_add_to_cart_order_mean', 'user_aisle_reorder_ratio', 'user_aisle_reorder_times', 'user_aisle_days_since_prior_order_mean', 'user_aisle_order_hour_of_day_mean', 'user_aisle_order_number_min', 'user_aisle_order_number_max', 'user_department_add_to_cart_order_max', 'user_department_add_to_cart_order_mean', 'user_department_reorder_times', 'user_department_reorder_ratio', 'user_department_days_since_prior_order_mean', 'user_department_order_hour_of_day_mean', 'user_department_order_dow_mean', 'user_add_to_cart_order_mean', 'user_product_count', 'user_reorder_times', 'user_reorder_ratio', 'user_order_hour_of_day_mean', 'user_order_dow_mean', 'user_days_since_prior_order_mean', 'product_add_to_cart_order_mean', 'product_reorder_ratio', 'product_reorder_times', 'product_order_hour_of_day_mean', 'product_order_dow_mean', 'product_days_since_prior_order_mean', 'user_product/user_aisle_bought_times', 'user_product/user_aisle_reorder_times', 'user_product/user_aisle_reorder_ratio', 'user_product/user_aisle_add_to_cart_order_min', 'user_product/user_aisle_add_to_cart_order_mean', 'user_product/user_aisle_order_number_max', 'user_product/user_aisle_order_number_min', 'user_product/user_department_bought_times', 'user_product/user_department_reorder_times', 'user_product/user_department_reorder_ratio', 'user_product/user_department_add_to_cart_order_mean', 'user_product/user_department_order_number_max', 'user_product/user_department_order_number_min', 'product/aisle_bought_times', 'product/aisle_reorder_ratio', 'product/aisle_add_to_cart_order_mean', 'product/department_reorder_times', 'product/department_reorder_ratio', 'product/department_add_to_cart_order_mean', 'user_product_bought_times/user_bought_times', 'user_product_bought_times/user_order_count', 'user_aisle_bought_times/user_order_count', 'user_department_bought_times/user_bought_times', 'user_department_bought_times/user_order_count', 'user_product_order_number_max/user_order_number_max', 'user_aisle_order_number_max/user_order_number_max', 'user_department_order_number_max/user_order_number_max', 'user_product_add_to_cart_order_mean/user_order_size_mean', 'user_product_add_to_cart_order_min/user_order_size_mean', 'user_aisle_add_to_cart_order_mean/user_order_size_mean', 'user_aisle_add_to_cart_order_min/user_order_size_mean', 'user_department_add_to_cart_order_mean/user_order_size_mean', 'user_product_reorder_ratio/user_reorder_ratio', 'user_aisle_reorder_ratio/user_reorder_ratio', 'user_department_reorder_ratio/user_reorder_ratio', 'user_product_bought_times/user_product_order_number_max-user_product_order_number_min', 'user_product_days_since_prior_order/user_product_days_since_prior_order_min', 'user_product_days_since_prior_order/user_product_days_since_prior_order_mean', 'user_bought_times/user_active_days', 'user_order_count/user_active_days', 'order_number-user_product_order_number_max', 'order_number-user_aisle_order_number_max', 'order_number-user_department_order_number_max', 'order_streak', 'user_active_days', 'user_order_days_since_prior_median', 'last', 'prev1', 'prev2', 'aisle_products', 'dep_products', 'up_order_rate_since_first_order', 'add_to_cart_order_inverted_mean', 'add_to_cart_order_relative_mean', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31']


@utils.timer
def lgbm_predict(model_file, input_file, output_file, is_test_set=False):
	import train_model_2
	df_predict, df_result = train_model_2.get_predict_file(input_file, is_test_set)
	print 'df_predict shape', df_predict.shape
	if is_test_set:
		print 'val set len', len(train_set_features)
		df_predict = df_predict[train_set_features]

	model = utils.load_model(model_file)
	print 'lgbm_predict best_iteration: ', model.best_iteration
	if model.best_iteration > 0:
		df_result['proba'] = model.predict(df_predict, num_iteration=model.best_iteration)
	else:
		df_result['proba'] = model.predict(df_predict, num_iteration=756)
	df_result = df_result.sort_values('proba', ascending=False).reset_index(drop=True)
	# print 'logloss: {}'.format(metrics.log_loss(df_result[LABEL], df_result['proba']))
	df_result.to_pickle(output_file)


@utils.timer
def convert_to_submit(predict_path, proba_threshold, output_name):
	df_order = pd.read_csv(ORDER_PATH)
	df_order = df_order[df_order['eval_set'] == 'test'][[UID, OID]]

	# df_order = df_order[df_order['eval_set'] == 'train'][[UID, OID]]
	# df_validate = pd.read_csv('../input/validate_sample.csv')
	# df_order = df_order[df_order[UID].isin(df_validate[UID])]

	df_predict = pd.read_csv(predict_path)

	df_predict = df_predict[df_predict['proba'] >= proba_threshold]
	df_predict = df_predict.groupby(UID)[PID].apply(lambda pids : ' '.join([str(pid) for pid in pids])).reset_index()

	df_order = df_order.merge(df_predict, how='left', on=[UID]).fillna('None').rename(columns={PID: 'products'})
	df_order[[OID, 'products']].to_csv(OUTPUT_DIR+output_name, index=False)


def run_predict():
	# name = 'XGBClassifier_colsample_bytree:0.8_learning_rate:0.1_max_depth:8_n_estimators:150_seed:0_subsample:0.8.plk'
	# name = 'XGBClassifier_colsample_bytree:0.8_learning_rate:0.1_max_depth:8_n_estimators:150_seed:0_subsample:0.8_07-17-00-40-20.plk'
	# name = 'XGBClassifier_colsample_bytree:0.8_learning_rate:0.1_max_depth:8_n_estimators:150_scale_pos_weight:2_seed:0_subsample:0.8_07-18-21-51-19.pkl'
	name = 'XGBClassifier_colsample_bytree:0.8_learning_rate:0.1_max_depth:8_n_estimators:150_seed:0_subsample:0.8_07-20-23-56-07.pkl'
	# name = 'XGBClassifier_colsample_bytree:0.8_learning_rate:0.1_max_depth:10_n_estimators:150_scale_pos_weight:5.0_seed:0_subsample:0.8_07-22-14-05-55.pkl'
	# name = 'XGBClassifier_colsample_bytree:0.8_learning_rate:0.1_max_depth:8_n_estimators:150_scale_pos_weight:5.0_seed:0_subsample:0.8_07-22-18-22-57.pkl'

	# name = 'Stacker:LogisticRegression_07-23-20-40-05.pkl'

	model = utils.load_model(name)
	# predict([model], INPUT_DIR+'test_diff.csv', PREDICT_DIR+'test_diff.csv')
	# predict([model], INPUT_DIR+'test_diff.csv', PREDICT_DIR+'test_diff_stack.csv')
	predict([model], INPUT_DIR+'validate_diff.csv', PREDICT_DIR+'validate_diff_xgb.csv')

	# test
	# predict([model], INPUT_DIR+'validate_sample_diff.csv', PREDICT_DIR+'validate_sample_diff.csv')


def run_lgbm_predict():
	# name = 'Booster_bagging_fraction:0.85_bagging_freq:3_categorical_column:[0, 1, 2]_feature_fraction:0.7_learning_rate:0.05_max_bin:255_max_depth:8_num_leaves:200_08-13-09-05-31.pkl'
	# name = 'Booster_bagging_fraction:0.85_bagging_freq:3_feature_fraction:0.7_learning_rate:0.04_max_bin:255_max_depth:8_num_leaves:200_08-13-10-54-44.pkl'
	# name = 'Booster_bagging_fraction:0.85_bagging_freq:3_feature_fraction:0.7_learning_rate:0.04_max_bin:255_max_depth:8_num_leaves:200_08-13-10-54-44.pkl'
	# name = 'Booster_bagging_fraction:0.85_bagging_freq:3_categorical_column:[0, 1, 2]_feature_fraction:0.7_learning_rate:0.04_max_bin:255_max_depth:8_num_leaves:200_08-13-12-16-48.pkl'
	name = 'Booster_bagging_fraction:0.85_bagging_freq:3_feature_fraction:0.7_learning_rate:0.04_max_bin:255_max_depth:8_num_leaves:170_08-14-01-05-21.pkl'
	predict_file = PREDICT_DIR+'validate_core_lgbm.pkl'
	# model = utils.load_model(name)
	# print '[lh] run_lgbm_predict'
	# print model.predict(pd.DataFrame({'aaaaaa': [1, 2, 3, 4]}))
	lgbm_predict(name, INPUT_DIR+'validate_diff_order_streaks_fix2_add_new.csv_core.csv', predict_file, False)
	validate_model_mean_f1(predict_file, False)


def check_train_set_feature(train_set_file):
	print 'check_train_set_feature', train_set_file
	import train_model_2
	df_predict, df_result = train_model_2.get_predict_file(train_set_file, is_train_set=True)
	cols = list(df_predict.columns)
	print 'train set features', cols
	for a, b in zip(cols, train_set_features):
		if a != b:
			print a, b
	# print cols
	# print 'col len {}'
	# pass


def run_lgbm_predict_submit():
	# name = 'Booster_bagging_fraction:0.85_bagging_freq:3_categorical_column:[0, 1, 2]_feature_fraction:0.7_learning_rate:0.04_max_bin:255_max_depth:8_num_leaves:200_08-13-12-16-48.pkl'
	# name = 'Booster_bagging_fraction:0.85_bagging_freq:3_categorical_column:[0, 1, 2]_feature_fraction:0.7_learning_rate:0.04_max_bin:255_max_depth:8_num_leaves:170_08-14-08-39-57.pkl'
	name = 'Booster_bagging_fraction:0.85_bagging_freq:3_categorical_column:[0, 1, 2]_feature_fraction:0.7_learning_rate:0.03_max_bin:255_max_depth:8_num_leaves:170_08-15-00-36-41.pkl'
	# name = 'Booster_bagging_fraction:0.85_bagging_freq:3_categorical_column:[0, 1, 2]_feature_fraction:0.6_learning_rate:0.03_max_bin:255_max_depth:8_num_leaves:170_08-15-02-58-34.pkl'
	predict_file = PREDICT_DIR + 'test_core_lgbm_submit.pkl'
	lgbm_predict(name, INPUT_DIR + 'test_diff_order_streaks_fix2_add_new.csv_core.csv', predict_file, True)
	convert_to_submit_max_f1(predict_file, 'submit_max_f1_' + utils.get_now_str() + '.csv')


# def test_lgbm_predict_submit():
# 	name = 'Booster_bagging_fraction:0.85_bagging_freq:3_categorical_column:[0, 1, 2]_feature_fraction:0.7_learning_rate:0.04_max_bin:255_max_depth:8_num_leaves:200_08-13-12-16-48.pkl'
# 	import train_model_2
# 	df_predict, df_result = train_model_2.get_predict_file(input_file, is_test_set)
# 	model = utils.load_model(model_file)
# 	if model.best_iteration > 0:
# 		df_result['proba'] = model.predict(df_predict, num_iteration=model.best_iteration)
# 	else:
# 		df_result['proba'] = model.predict(df_predict)
# 	df_result = df_result.sort_values('proba', ascending=False).reset_index(drop=True)
# 	df_result.to_pickle(output_file)


def run_convert():
	# proba_threshold = 0.53480
	# proba_threshold = 0.56982
	# proba_threshold = 0.70710
	# proba_threshold = 0.53781
	# proba_threshold = 0.70056
	# proba_threshold = 0.68611
	# proba_threshold = 0.55909
	proba_threshold = 0.5

	convert_to_submit(PREDICT_DIR+'test_diff_stack.csv', proba_threshold, 'submit_' + utils.get_now_str() + '.csv')


def apply_f_score(series):
	return utils.f1_score_single(series['true_products'], series['products'])
	# print '555555', type(series), series
	# return 1


@utils.timer
def validate_model_mean_f1(predict_file, save_analysis=False):
	import max_f1_predict
	begin_time = time.time()

	# df = pd.read_csv(PREDICT_DIR + predict_file)
	df = pd.read_pickle(PREDICT_DIR + predict_file)

	# df_validate = pd.read_csv(INPUT_DIR+ground_truth_file)
	# print df_validate[UID].nunique()
	df_positive = df[df.label == 1]
	# print df_positive[UID].nunique()
	df_positive = df_positive.groupby(UID)[PID].apply(lambda pids:list(pids)).reset_index().rename(columns={PID: 'true_products'})

	# df = pd.read_csv(PREDICT_DIR + predict_file)
	df_users = pd.DataFrame({UID: df[UID].unique()})
	df_users = df_users.merge(df_positive, how='left', on=[UID])
	for index in df_users[df_users['true_products'].isnull()].index:
		df_users.ix[index, 'true_products'] = [None]

	print 'step 1 elapsed: {}'.format(time.time()-begin_time)

	# df_pred = df.groupby(UID).apply(max_f1_predict.get_best_prediction_group).reset_index()
	df_pred = df.groupby(UID).apply(max_f1_predict.get_best_prediction_group_submit).reset_index()
	# print df_pred.head()

	df_users = df_users.merge(df_pred, how='left', on=[UID])

	df_users['scores'] = df_users.apply(apply_f_score, axis=1)

	if save_analysis:
		df_users.to_csv(PREDICT_DIR+'_'.join(['analysis', utils.get_now_str(), predict_file]), index=False)

	print 'mean_f_score: {}'.format(df_users['scores'].mean())


def list_str_eval(list_str):
	elems = eval(list_str)
	return [eval(elem) if isinstance(elem, str) else elem for elem in elems]


def read_max_f1_analysis(input_file):
	df_result = pd.read_csv(input_file)
	df_result['true_products'] = df_result['true_products'].apply(list_str_eval)
	df_result['products'] = df_result['products'].apply(list_str_eval)
	return df_result


def get_true_product_pred_proba(user_id, products, df_predict):
	df_predict = df_predict[df_predict[UID] == user_id]
	df = df_predict[df_predict[PID].isin(products)] if products else df_predict
	return df.sort_values('proba', ascending=False)


@utils.timer
def convert_to_submit_max_f1(predict_file, output_file):
	import max_f1_predict

	df_order = pd.read_csv(ORDER_PATH)
	df_predict = pd.read_pickle(predict_file)
	df_order = df_order[df_order['eval_set'] == 'test'][[UID, OID]]

	# df_order = df_order[df_order['eval_set'] == 'train'][[UID, OID]]
	# df_order = df_order[df_order[UID].isin(df_predict[UID])]

	df_predict = df_predict.groupby(UID).apply(max_f1_predict.get_best_prediction_group_submit).reset_index()
	df_predict['products'] = df_predict['products'].apply(lambda pids: ' '.join([str(pid) for pid in pids]))
	df_order = df_order.merge(df_predict, how='left', on=[UID])
	df_order[[OID, 'products']].to_csv(OUTPUT_DIR + output_file, index=False)
	# df_order[[UID, OID, 'products']].to_csv(OUTPUT_DIR + output_file, index=False)


def run_validate_model_mean_f1():
	# reorder_size_file = REORDER_INPUT_DIR + 'train_true_reorder_size.csv'
	# test
	# validate_model_mean_f1('validate_sample_diff.csv', 'validate_sample_diff.csv', 0.4, False)

	# reorder_size_file = PREDICT_DIR + 'predict_validate_reorder_size.csv'
	# validate_model_mean_f1('validate_diff_stack.csv', 'validate_diff.csv', reorder_size_file, 0.5, False)
	# validate_model_mean_f1('validate_diff_xgb.csv', 'validate_diff.csv', reorder_size_file, 0.48, False)

	# validate_model_mean_f1('validate_diff_stack.csv', 'validate_diff.csv', 0.46, False)
	# validate_model_mean_f1('validate_diff_stack.csv', 'validate_diff.csv', 0.44, False)
	# validate_model_mean_f1('validate_core_lgbm.pkl', True)
	validate_model_mean_f1('validate_core_lgbm_08-13-11-00-07.pkl', True)


def run_convert_to_submit_max_f1():
	# convert_to_submit_max_f1(PREDICT_DIR+'test_diff_stack', 'submit_max_f1' + utils.get_now_str() + '.csv')
	convert_to_submit_max_f1(PREDICT_DIR+'validate_core_lgbm_08-13-11-00-07.pkl', 'submit_max_f1_' + utils.get_now_str() + '.csv')


def check_result():
	pass

if __name__ == '__main__':
	pass
	# run_predict()
	# run_convert()
	# run_validate_model_mean_f1()

	# run_convert_to_submit_max_f1()

	# run_lgbm_predict()
	# run_convert_to_submit_max_f1()
	run_lgbm_predict_submit()

	# df1 = pd.read_csv(PREDICT_DIR + 'validate_diff_stack.csv')
	# df2 = pd.read_csv(PREDICT_DIR + 'predict_validate_reorder_size.csv')
	# print df1[df1[UID].isin(df2[UID])][UID].shape, df2.shape, df1.shape

	# import train_model_2
	# df_predict, df_result = train_model_2.get_predict_file(INPUT_DIR+'validate_diff_order_streaks_fix_add_new_core.csv')
	# print df_result.dtypes
	# print df_predict[[AID, DID]].dtypes

	# df = pd.read_csv(OUTPUT_DIR+'submit_max_f108-13-13-20-12.csv')
	# print df.head(10)

	# save_col_name(INPUT_DIR+'validate_diff_order_streaks_fix_add_new_core.csv')
	# check_train_set_feature(INPUT_DIR+'train_diff_order_streaks_fix_add_new_core.csv')
	# check_train_set_feature(INPUT_DIR+'train_diff_order_streaks_fix2_add_new.csv_core.csv')
	# run_lgbm_predict()