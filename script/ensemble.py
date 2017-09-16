# -*- coding: utf-8 -*-

import time
import random
import gc

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, KFold, GridSearchCV
from sklearn import ensemble, linear_model
from sklearn.base import clone
import xgboost as xgb

import utils
import train_model
from const import *


np.random.seed(RAND_SEED)
random.seed(RAND_SEED)


class EnsembleClassifier(object):

	def __init__(self, n_folds, stacker, base_models, param_grid, grid_cv=3):
		self.n_folds = n_folds
		self.base_models = base_models
		self.trained_based_models = []

		self.param_grid = param_grid
		self.stacker = stacker
		self.stacker_grid_cv = None
		self.grid_cv = grid_cv

	def fit(self, X, y):
		"""
		:param X: DataFrame
		:param y: Series
		:return:
		"""
		start_time = time.time()

		# to save memory
		# X = np.array(X)
		# y = np.array(y)

		sfk = StratifiedKFold(self.n_folds, random_state=RAND_SEED)
		# S_train = np.zeros((X.shape[0], len(self.base_models)))
		S_train = pd.DataFrame(
			0, index=np.arange(X.shape[0]),
			columns=[base_model.__class__.__name__ for base_model in self.base_models])

		for i, clf in enumerate(self.base_models):
			print 'Fitting For Base Model #{} / {} ---'.format(i+1, len(self.base_models))
			models = []
			for j, (train_idx, test_idx) in enumerate(sfk.split(X, y)):
				print '--- Fitting For Fold {} / {} ---'.format(j+1, self.n_folds)
				print 'gc before train', gc.collect()

				X_train = X.iloc[train_idx]
				y_train = y.iloc[train_idx]
				X_holdout = X.iloc[test_idx]
				# y_holdout = y[test_idx]
				clone_clf = clone(clf)
				clone_clf.fit(X_train, y_train)
				models.append(clone_clf)
				y_pred = clone_clf.predict_proba(X_holdout)[:, 1]
				S_train.iloc[test_idx, i] = y_pred

				print 'gc after train', gc.collect()

				print 'Elapsed: {} minutes ---'.format(round(((time.time() - start_time) / 60), 2))

			self.trained_based_models.append(models)
			print 'Elapsed: {} minutes ---'.format(round(((time.time() - start_time) / 60), 2))

		print '--- Base Models Trained: {} minutes ---'.format(round(((time.time() - start_time) / 60), 2))

		print 'Base Models Prediction Corr'
		print S_train.corr()
		self._save_stacker_train_set(S_train)

		self._fit_stacker(S_train, y, start_time)

	def _fit_stacker(self, S_train, y, start_time):
		grid = GridSearchCV(estimator=self.stacker, param_grid=self.param_grid, n_jobs=-1, cv=self.grid_cv, scoring='roc_auc')
		grid.fit(S_train, y)
		self.stacker_grid_cv = grid

		try:
			print 'Fold num:', self.n_folds
			print 'Param grid:', self.param_grid
			print 'Best Params:', grid.best_params_
			print 'Best CV Score:', grid.best_score_
			print 'Best estimator:', grid.best_estimator_
		except:
			pass

		print '--- Stacker Trained: {} minutes ---'.format(round(((time.time() - start_time) / 60), 2))

	@staticmethod
	def _save_stacker_train_set(S_train):
		S_train_name = 'stacker_train' + '_' + utils.get_now_str() + '.csv'
		S_train.to_csv(STACKING_DIR + S_train_name, index=False)
		print 'save stacker train set:', S_train_name

	def fit_stacker(self, X, y, ensemble_model, stacker_train_set=None):
		"""
		make sure ensemble_model is trained on X, y
		:param X:
		:param y:
		:param ensemble_model: EnsembleClassifier
		:param stacker_train_set: Stacker train set. use to speed up train stack
		:return:
		"""
		start_time = time.time()

		self.n_folds = ensemble_model.n_folds
		self.base_models = ensemble_model.base_models
		self.trained_based_models = ensemble_model.trained_based_models

		if stacker_train_set is not None:
			S_train = stacker_train_set
		else:
			sfk = StratifiedKFold(self.n_folds, random_state=RAND_SEED)

			S_train = pd.DataFrame(
				0, index=np.arange(X.shape[0]),
				columns=[base_model.__class__.__name__ for base_model in self.base_models])

			for i, clf in enumerate(self.base_models):
				print 'Fitting For Base Model #{} / {} ---'.format(i+1, len(self.base_models))

				for j, (train_idx, test_idx) in enumerate(sfk.split(X, y)):
					print '--- Fitting For Fold {} / {} ---'.format(j+1, self.n_folds)
					X_holdout = X.iloc[test_idx]
					y_pred = self.trained_based_models[i][j].predict_proba(X_holdout)[:, 1]
					S_train.iloc[test_idx, i] = y_pred
					print 'gc after train', gc.collect()

					print 'Elapsed: {} minutes ---'.format(round(((time.time() - start_time) / 60), 2))

				print 'Elapsed: {} minutes ---'.format(round(((time.time() - start_time) / 60), 2))

			print '--- Base Models Trained: {} minutes ---'.format(round(((time.time() - start_time) / 60), 2))

			# print 'Base Models Prediction Corr'
			# print S_train.corr()
			self._save_stacker_train_set(S_train)

		self._fit_stacker(S_train, y, start_time)

	def predict_proba(self, X):
		"""
		:param X: DataFrame
		:return:
		"""
		# to save memory
		# X = np.array(X)

		S_test = np.zeros((X.shape[0], len(self.base_models)))

		for i, clf in enumerate(self.base_models):
			S_test_i = np.zeros((X.shape[0], self.n_folds))
			for j in xrange(self.n_folds):
				S_test_i[:, j] = self.trained_based_models[i][j].predict_proba(X)[:, 1]

			S_test[:, i] = S_test_i.mean(1)

		return self.stacker_grid_cv.predict_proba(S_test)

	@property
	def name(self):
		return 'Stacker:'+self.stacker.__class__.__name__


def test_EnsembleClassifier():
	pass
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

	base_models = [
		ensemble.RandomForestClassifier(n_estimators=20, random_state=RAND_SEED),
		xgb.XGBClassifier(n_estimators=20, seed=RAND_SEED)
	]

	ensembleClf = EnsembleClassifier(
		n_folds=2,
		stacker=ensemble.ExtraTreesClassifier(random_state=RAND_SEED),
		base_models=base_models,
		param_grid=dict(n_estimators=[10, 20])
	)

	ensembleClf.fit(X_train, y_train)

	train_proba = ensembleClf.predict_proba(X_train)[:, 1]
	train_auc = roc_auc_score(y_train, train_proba)

	validate_proba = ensembleClf.predict_proba(X_validate)[:, 1]
	validate_auc = roc_auc_score(y_validate, validate_proba)

	print train_auc, validate_auc

	model_name = utils.save_model(ensembleClf, {})

	load_model = utils.load_model(model_name)

	validate_proba = load_model.predict_proba(X_validate)[:, 1]
	validate_auc = roc_auc_score(y_validate, validate_proba)
	print validate_auc


def run(train_path, validate_path):
	from sklearn.metrics import roc_auc_score

	scale_pos_weight = 2
	X_train, y_train = train_model.get_train_set(train_path, scale_pos_weight)
	gc.collect()

	X_validate, Y_validate = train_model.get_validate_set(validate_path)
	gc.collect()

	base_models = [
		xgb.XGBClassifier(
			n_estimators=150, learning_rate=0.1, subsample=0.8,
			max_depth=8, colsample_bytree=0.8, seed=RAND_SEED,
		),
		ensemble.GradientBoostingClassifier(
			n_estimators=150, learning_rate=0.1, subsample=0.8,
			max_depth=8, max_features=0.1, random_state=RAND_SEED,
		),
		ensemble.RandomForestClassifier(
			n_estimators=200, max_depth=13, max_features='sqrt',
			# min_samples_split=15, min_samples_leaf=10,
			random_state=RAND_SEED, n_jobs=-1,
		),
		ensemble.ExtraTreesClassifier(
			n_estimators=200, max_depth=13, max_features='sqrt',
			# min_samples_split=15, min_samples_leaf=10,
			random_state=RAND_SEED, n_jobs=-1,
		),
	]

	stacker = linear_model.LogisticRegression()
	param_grid = dict(C=[0.0001, 0.001, 0.01, 0.1])

	# stacker = xgb.XGBClassifier(seed=RAND_SEED)
	# param_grid = dict()

	ensembleClf = EnsembleClassifier(
		n_folds=3,
		stacker=stacker,
		base_models=base_models,
		param_grid=param_grid,
		grid_cv=5,
	)

	save_model = True

	ensembleClf.fit(X_train, y_train)
	gc.collect()
	save_model_name = utils.save_model(ensembleClf, model_name=ensembleClf.name) if save_model else None
	print 'save_model_name:', save_model_name

	train_proba = ensembleClf.predict_proba(X_train)[:, 1]
	gc.collect()
	train_auc = roc_auc_score(y_train, train_proba)
	print 'train auc: {}'.format(train_auc)

	validate_proba = ensembleClf.predict_proba(X_validate)[:, 1]
	validate_auc = roc_auc_score(Y_validate.label, validate_proba)
	print 'validate auc: {}'.format(validate_auc)


@utils.timer
def train_stacker(train_path, validate_path):
	from sklearn.metrics import roc_auc_score

	scale_pos_weight = 2
	X_train, y_train = train_model.get_train_set(train_path, scale_pos_weight)
	gc.collect()

	X_validate, Y_validate = train_model.get_validate_set(validate_path)
	gc.collect()

	# stacker = linear_model.LogisticRegression(random_state=RAND_SEED)
	# param_grid = dict(C=[0.5, 1, 5])

	stacker = xgb.XGBClassifier(
		n_estimators=100,
		learning_rate=0.1,
		# max_depth=8,
		subsample=0.8,
		colsample_bytree=0.8,
		seed=RAND_SEED,
	)
	param_grid = dict(max_depth=[7, 8, 9])

	ensembleClf = EnsembleClassifier(
		stacker=stacker,
		param_grid=param_grid,
		grid_cv=5,
		n_folds=None,
		base_models=None,
	)

	source_model_name = 'Stacker:LogisticRegression_07-23-20-40-05.pkl'
	stacker_train_set = pd.read_csv(STACKING_DIR+'stacker_train_07-24-01-20-40.csv')
	# stacker_train_set = None
	save_model = False

	sourceEnsembleClf = utils.load_model(source_model_name)
	ensembleClf.fit_stacker(X_train, y_train, sourceEnsembleClf, stacker_train_set)

	save_model_name = utils.save_model(ensembleClf, model_name=ensembleClf.name) if save_model else None
	print 'save_model_name:', save_model_name
	#
	# train_proba = ensembleClf.predict_proba(X_train)[:, 1]
	# gc.collect()
	# train_auc = roc_auc_score(y_train, train_proba)
	# print 'train auc: {}'.format(train_auc)
	#
	# validate_proba = ensembleClf.predict_proba(X_validate)[:, 1]
	# validate_auc = roc_auc_score(Y_validate.label, validate_proba)
	# print 'validate auc: {}'.format(validate_auc)


def some_test():
	pass


def run_train_stacker():
	# train_stacker(INPUT_DIR+'train_sample_ratio.csv', INPUT_DIR+'validate_sample_ratio.csv')
	train_stacker(INPUT_DIR+'train_diff.csv', INPUT_DIR+'validate_diff.csv')


def test_run():
	run(INPUT_DIR+'train_sample_ratio.csv', INPUT_DIR+'validate_sample_ratio.csv')


def formal_run():
	run(INPUT_DIR + 'train_diff.csv', INPUT_DIR + 'validate_diff.csv')

if __name__ == '__main__':
	pass
	# test_EnsembleClassifier()

	# test_run()
	# formal_run()
	train_model.tune_positive()
	# run_train_stacker()