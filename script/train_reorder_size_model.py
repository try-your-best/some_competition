# -*- coding: utf-8 -*-

import xgboost as xgb
from sklearn import model_selection, linear_model, metrics

from const import *
import utils

np.random.seed(RAND_SEED)
random.seed(RAND_SEED)


def get_train_and_validate_data(train_file, validate_file):
	df_train = pd.read_csv(train_file)

	df_train = utils.preprocess_reorder_size_data(df_train)
	X_train = df_train.drop(['reorder_size'], axis=1)
	y_train = df_train['reorder_size']

	del df_train
	gc.collect()

	df_validate = pd.read_csv(validate_file)
	Y_validate = df_validate[[UID, 'reorder_size']]

	df_validate = utils.preprocess_reorder_size_data(df_validate)
	X_validate = df_validate.drop(['reorder_size'], axis=1)

	del df_validate
	gc.collect()

	print 'train columns', list(X_train.columns)
	return X_train, y_train, X_validate, Y_validate


@utils.timer
def run(train_file, validate_file):
	X_train, y_train, X_validate, Y_validate = get_train_and_validate_data(train_file, validate_file)

	estimator = xgb.XGBRegressor(seed=RAND_SEED)
	param_grid = dict(
		n_estimators=[37],
		learning_rate=[0.08],
		max_depth=[7],
		subsample=[0.8],
		colsample_bytree=[1],
	)

	# estimator = linear_model.Ridge(random_state=RAND_SEED)
	# estimator = linear_model.Lasso(random_state=RAND_SEED)
	# param_grid = dict(
	# 	alpha=[0.01, 0.1, 1],
	# 	normalize=[True, False]
	# )

	scorer = metrics.make_scorer(score_func, False)
	k_fold = model_selection.KFold(5, random_state=RAND_SEED)
	grid_cv = model_selection.GridSearchCV(estimator, param_grid=param_grid, cv=k_fold, scoring=scorer)
	grid_cv.fit(X_train, y_train)

	print 'Best Params:', grid_cv.best_params_
	print 'Best CV Score:', grid_cv.best_score_

	print 'validate score: {}'.format(grid_cv.score(X_validate, Y_validate['reorder_size']))

	y_preds = grid_cv.predict(X_validate)
	Y_validate['pred_reorder_size_float'] = y_preds
	Y_validate['pred_reorder_size'] = process_pred(y_preds)

	Y_validate.to_csv(PREDICT_DIR+'predict_validate_reorder_size.csv', index=False)

	return grid_cv


def process_pred(preds):
	preds[preds < 0] = 0
	return np.around(preds)


def eval_func(preds, dtrain):
	preds = process_pred(preds)
	labels = dtrain.get_label()
	return 'error', metrics.mean_absolute_error(labels, preds)


def score_func(y_true, y_pred):
	y_pred = process_pred(y_pred)
	return metrics.mean_absolute_error(y_true, y_pred)


def run_cv(train_file, validate_file):
	X_train, y_train, X_validate, Y_validate = get_train_and_validate_data(train_file, validate_file)
	param = dict(eta=0.08, max_depth=7, subsample=0.8, colsample_bytree=1, objective='reg:linear', silent=1)

	dtrain = xgb.DMatrix(X_train.values, y_train.values)

	print type(dtrain.get_label())

	return
	res = xgb.cv(param, dtrain, num_boost_round=100, nfold=5, feval=eval_func, seed=RAND_SEED,
		callbacks=[xgb.callback.print_evaluation(show_stdv=True), xgb.callback.early_stop(15)])

	print res


if __name__ == '__main__':
	pass
	# run(REORDER_INPUT_DIR+'train.csv', REORDER_INPUT_DIR+'validate.csv')

	# run(REORDER_INPUT_DIR+'train_all_reorder.csv', REORDER_INPUT_DIR+'validate_all_reorder.csv')


	run_cv(REORDER_INPUT_DIR+'train_all_reorder.csv', REORDER_INPUT_DIR+'validate_all_reorder.csv')
