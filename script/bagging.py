# -*- coding: utf-8 -*-

from imblearn.ensemble import EasyEnsemble
from sklearn.base import clone
from sklearn import metrics
import xgboost as xgb

import utils
from const import *
import train_none_order_model

np.random.seed(RAND_SEED)
random.seed(RAND_SEED)


def bag_model(X, y, ):
	pass


class BagClassifier(object):

	def __init__(self, base_model, n_subsets):
		self.base_model = base_model
		self.n_subsets = n_subsets
		self.easy_ensemble = EasyEnsemble('auto', random_state=RAND_SEED, n_subsets=4)
		self.trained_based_models = []

	def fit(self, X, y):
		X_s, y_s = self.easy_ensemble.fit_sample(X, y)
		for idx in xrange(self.n_subsets):
			clone_model = clone(self.base_model)
			clone_model.fit(X_s[idx], y_s[idx])
			self.trained_based_models.append(clone_model)

	def predict_proba(self, X):
		S_test = np.zeros((X.shape[0], len(self.trained_based_models)))
		for idx, clf in enumerate(self.trained_based_models):
			S_test[:, idx] = clf.predict_proba(X)[:, 1]
		return S_test.mean(1)


def run_bagging(train_file, validate_file):
	X_train, y_train, X_validate, Y_validate = train_none_order_model.get_train_and_validate_data(train_file, validate_file)
	# print (y_train == 1).sum(), y_train.shape
	# easy_ensemble = EasyEnsemble('auto', random_state=RAND_SEED, n_subsets=4)
	# X_res, y_res = easy_ensemble.fit_sample(X_train, y_train)
	# print y_res.shape
	# print (y_res[0] == 1).sum()
	# print X_res.shape
	# print X_res

	param = dict(max_depth=4, learning_rate=0.03, n_estimators=335,
		subsample=0.9, colsample_bytree=0.85,
		# scale_pos_weight=scale_pos_weight
	)
	base_model = xgb.XGBClassifier(seed=RAND_SEED, **param)
	bag = BagClassifier(base_model, 4)
	bag.fit(X_train, y_train)

	y_train_pred = bag.predict_proba(X_train.values)
	print 'train auc: ', metrics.roc_auc_score(y_train, y_train_pred)
	y_validate_pred = bag.predict_proba(X_validate.values)
	print 'validate auc: ', metrics.roc_auc_score(Y_validate['label'], y_validate_pred)

	Y_validate['proba'] = y_validate_pred
	Y_validate = Y_validate.sort_values('proba', ascending=False).reset_index(drop=True)
	Y_validate.to_csv(PREDICT_DIR + 'validate_none_order_proba_bag.csv', index=False)


if __name__ == '__main__':
	run_bagging(NONE_ORDER_INPUT_DIR+'train_ratio.csv', NONE_ORDER_INPUT_DIR + 'validate_ratio.csv')