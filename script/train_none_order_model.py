# -*- coding: utf-8 -*-

import xgboost as xgb
from sklearn import model_selection, linear_model, metrics

from const import *
import utils
import train_model

np.random.seed(RAND_SEED)
random.seed(RAND_SEED)


def get_train_and_validate_data(train_file, validate_file, negative_ratio=None):
	df_train = pd.read_csv(train_file)

	if negative_ratio:
		positive = df_train[df_train.label == 1]
		negative = df_train[df_train.label == 0].sample(positive.shape[0] * negative_ratio)
		df_train = utils.concat([positive, negative]).sample(frac=1).reset_index(drop=True)

	df_train = utils.preprocess_none_order_data(df_train)
	X_train = df_train.drop(['label'], axis=1)
	y_train = df_train['label']

	del df_train
	gc.collect()

	df_validate = pd.read_csv(validate_file)
	Y_validate = df_validate[[UID, 'label']]

	df_validate = utils.preprocess_none_order_data(df_validate)
	X_validate = df_validate.drop(['label'], axis=1)

	del df_validate
	gc.collect()

	print 'train columns', len(X_train.columns)
	return X_train, y_train, X_validate, Y_validate


@utils.timer
def run_cv(train_file, validate_file):
	X_train, y_train, X_validate, Y_validate = get_train_and_validate_data(train_file, validate_file, 3)
	print 'train shape: {}, validate shape: {}'.format(X_train.shape, X_validate.shape)
	# scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
	# print 'scale_pos_weight: ', scale_pos_weight

	param = dict(eta=0.03, max_depth=4, subsample=0.9, colsample_bytree=0.85,
		# scale_pos_weight=scale_pos_weight,
		objective='binary:logistic', silent=1)

	dtrain = xgb.DMatrix(X_train.values, y_train.values)

	# k_fold = model_selection.StratifiedKFold(5, random_state=RAND_SEED)

	res = xgb.cv(param, dtrain, num_boost_round=400,
		nfold=5, stratified=True,
		# folds=k_fold.split(X_train, y_train),
		metrics='auc', seed=RAND_SEED,
		callbacks=[xgb.callback.print_evaluation(show_stdv=True), xgb.callback.early_stop(40)])

	print res


@utils.timer
def run_train(train_file, validate_file, save_model=False):
	X_train, y_train, X_validate, Y_validate = get_train_and_validate_data(train_file, validate_file)

	# scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
	param = dict(max_depth=4, learning_rate=0.024, n_estimators=400,
		subsample=0.9, colsample_bytree=0.85,
		# scale_pos_weight=scale_pos_weight
	)
	clf = xgb.XGBClassifier(seed=RAND_SEED, **param)
	clf.fit(X_train, y_train,
		eval_set=[(X_train, y_train), (X_validate, Y_validate['label'])],
		early_stopping_rounds=40, eval_metric='auc', verbose=True
	)
	# clf.fit(X_train, y_train, eval_set=[(X_train, y_train)], eval_metric='logloss', verbose=False)

	if save_model:
		save_model_name = utils.save_model(clf, param)
		print 'save model name: {}'.format(save_model_name)

	Y_validate['proba'] = clf.predict_proba(X_validate)[:, 1]
	Y_validate = Y_validate.sort_values('proba', ascending=False).reset_index(drop=True)
	# print Y_validate.head(20)

	Y_validate.to_csv(PREDICT_DIR+'validate_none_order_prob_ratio.csv', index=False)

	# print 'feature gain', '*' * 40
	# df_f_gain = utils.get_feature_score(clf._Booster.get_score(importance_type='gain'))
	# print df_f_gain
	#
	# print 'feature weight', '*' * 40
	# df_f_weight = utils.get_feature_score(clf._Booster.get_score(importance_type='weight'))
	# print df_f_weight
	#
	# print 'feature cover', '*' * 40
	# df_f_weight = utils.get_feature_score(clf._Booster.get_score(importance_type='cover'))
	# print df_f_weight

	return clf



def tune_f1(df_result, positiveTune):
	df_result = df_result.sort_values('proba', ascending=False).reset_index(drop=True)
	sample_len = df_result.shape[0]
	df_result['pred'] = np.zeros(sample_len)

	f1, precision, recall, proba_threshold, top_num = 0, 0, 0, 0, 0

	for num in xrange(positiveTune.start, positiveTune.end, positiveTune.step):
		df_result.ix[:num, 'pred'] = 1
		cur_f1, cur_precision, cur_recall = train_model.full_f1_score(df_result['label'], df_result['pred'])

		if cur_f1 > f1:
			f1 = cur_f1
			precision = cur_precision
			recall = cur_recall
			top_num = num
			proba_threshold = df_result['proba'][top_num - 1]

	print 'tune_f1 result: f1: {}, precision: {}, recall: {}, proba_threshold: {}, top_num: {}'.format(
		f1, precision, recall, proba_threshold, top_num)


def run_tune_f1():
	df_result = pd.read_csv(PREDICT_DIR+'validate_none_order_proba.csv')

	positive_num = df_result['label'].sum()
	start = positive_num
	end = int(positive_num * 4)
	positiveTune = train_model.PositiveTune(start, end, (end - start) / 100)
	tune_f1(df_result, positiveTune)


if __name__ == '__main__':
	pass

	# run_cv(NONE_ORDER_INPUT_DIR+'train.csv', NONE_ORDER_INPUT_DIR + 'validate.csv')

	run_cv(NONE_ORDER_INPUT_DIR + 'train_ratio.csv', NONE_ORDER_INPUT_DIR + 'validate_ratio.csv')

	# run_train(NONE_ORDER_INPUT_DIR+'train.csv', NONE_ORDER_INPUT_DIR + 'validate.csv', False)

	# run_train(NONE_ORDER_INPUT_DIR+'train_ratio.csv', NONE_ORDER_INPUT_DIR + 'validate_ratio.csv', False)

	# run_tune_f1()