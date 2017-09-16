# -*- coding: utf-8 -*-
"""
@author: Faron
"""
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from datetime import datetime

from const import *

'''
This kernel implements the O(n²) F1-Score expectation maximization algorithm presented in
"Ye, N., Chai, K., Lee, W., and Chieu, H.  Optimizing F-measures: A Tale of Two Approaches. In ICML, 2012."

It solves argmax_(0 <= k <= n,[[None]]) E[F1(P,k,[[None]])]
with [[None]] being the indicator for predicting label "None"
given posteriors P = [p_1, p_2, ... , p_n], where p_1 > p_2 > ... > p_n
under label independence assumption by means of dynamic programming in O(n²).
'''

WARNING_F1 = 0.4


class F1Optimizer():
	def __init__(self):
			pass

	@staticmethod
	def get_expectations(P, pNone=None):
			expectations = []
			P = np.sort(P)[::-1]

			n = np.array(P).shape[0]
			DP_C = np.zeros((n + 2, n + 1))
			if pNone is None:
				pNone = (1.0 - P).prod()

			DP_C[0][0] = 1.0

			# 处理 k1 == 0 的情况
			for j in range(1, n):
				DP_C[0][j] = (1.0 - P[j - 1]) * DP_C[0, j - 1]

			for i in range(1, n + 1):
				# 处理 k1 == k 的情况
				DP_C[i, i] = DP_C[i - 1, i - 1] * P[i - 1]
				# 处理 k1 < k 的情况
				for j in range(i + 1, n + 1):
						DP_C[i, j] = P[j - 1] * DP_C[i - 1, j - 1] + (1.0 - P[j - 1]) * DP_C[i, j - 1]

			DP_S = np.zeros((2 * n + 1,))
			DP_SNone = np.zeros((2 * n + 1,))
			for i in range(1, 2 * n + 1):
					# s(n, a) = 1/a
					DP_S[i] = 1. / (1. * i)
					# f1None + 2 * pNone / (2 + k), 看完整的公式， 经过提取，就知道 1. / (1. * i + 1) 的原因
					DP_SNone[i] = 1. / (1. * i + 1)
			for k in range(n + 1)[::-1]:
				f1 = 0
				f1None = 0
				for k1 in range(n + 1):
					f1 += 2 * k1 * DP_C[k1][k] * DP_S[k + k1]
					f1None += 2 * k1 * DP_C[k1][k] * DP_SNone[k + k1]
				for i in range(1, 2 * k - 1):
					# i 从 1 开始， 下一轮 k + k1 只会用到 (k-1) + (k-1) == 2*k-2
					# 注意， 顺序是从小到大， 因为 DP_S[i] 依赖到上一轮的 DP_S[i] 和 DP_S[i+1]
					DP_S[i] = (1 - P[k - 1]) * DP_S[i] + P[k - 1] * DP_S[i + 1]
					DP_SNone[i] = (1 - P[k - 1]) * DP_SNone[i] + P[k - 1] * DP_SNone[i + 1]
				# f1None + 2 * pNone / (2 + k) 每一个 k 都加上空单的概率
				expectations.append([f1None + 2 * pNone / (2 + k), f1])

			return np.array(expectations[::-1]).T

	@staticmethod
	def maximize_expectation(P, pNone=None):
			expectations = F1Optimizer.get_expectations(P, pNone)

			ix_max = np.unravel_index(expectations.argmax(), expectations.shape)
			max_f1 = expectations[ix_max]

			predNone = True if ix_max[0] == 0 else False
			best_k = ix_max[1]

			return best_k, predNone, max_f1

	@staticmethod
	def _F1(tp, fp, fn):
			return 2 * tp / (2 * tp + fp + fn)

	@staticmethod
	def _Fbeta(tp, fp, fn, beta=1.0):
			beta_squared = beta ** 2
			return (1.0 + beta_squared) * tp / ((1.0 + beta_squared) * tp + fp + beta_squared * fn)


def print_best_prediction(P, pNone=None):
	print("Maximize F1-Expectation")
	print("=" * 23)
	P = np.sort(P)[::-1]
	n = P.shape[0]
	L = ['L{}'.format(i + 1) for i in range(n)]

	if pNone is None:
			print("Estimate p(None|x) as (1-p_1)*(1-p_2)*...*(1-p_n)")
			pNone = (1.0 - P).prod()

	PL = ['p({}|x)={}'.format(l, p) for l, p in zip(L, P)]
	print("Posteriors: {} (n={})".format(PL, n))
	print("p(None|x)={}".format(pNone))

	opt = F1Optimizer.maximize_expectation(P, pNone)
	best_prediction = [None] if opt[1] else []
	best_prediction += (L[:opt[0]])
	f1_max = opt[2]

	print("Prediction {} yields best E[F1] of {}\n".format(best_prediction, f1_max))


def get_best_prediction_group(df):
	# df = df[df['proba'] >= min_positive_proba]
	# if df.empty:
	# 	# print 'empty best_prediction user_id {}'.format(user_id)
	# 	best_prediction = [None]
	# 	best_k_proba = [1]
	# else:
	# 	df = df.sort_values('proba', ascending=False).reset_index()
	# 	best_prediction, best_k_proba = get_best_prediction(df[PID], df['proba'], user_id)

	# more bold
	# df = df.sort_values('proba', ascending=False).reset_index()
	# df = df.head(df_reorder_size.ix[user_id, 'reorder_size'])

	user_id = df[UID].iloc[0]
	pNone = (1.0 - df['proba']).prod()
	# save some compute
	df = df[df['proba'] > 0.01]
	df = df.sort_values('proba', ascending=False).reset_index().head(80)
	best_prediction, best_k_proba = get_best_prediction(df[PID], df['proba'], user_id, pNone)
	return pd.Series([best_prediction, best_k_proba], index=['products', 'best_k_proba'])


def get_best_prediction(products, P, user_id, pNone=None):
	"""
	:param products:
	:param P: make sure P is descending
	:param pNone:
	:return:
	"""
	# print 'get_best_prediction', products.shape, P.shape
	products = list(products)
	# return [1]
	# print("Maximize F1-Expectation")
	# print("=" * 23)
	# P = np.sort(P)[::-1]
	# n = P.shape[0]
	# L = ['L{}'.format(i + 1) for i in range(n)]

	if pNone is None:
			# print("Estimate p(None|x) as (1-p_1)*(1-p_2)*...*(1-p_n)")
			pNone = (1.0 - P).prod()

	# PL = ['p({}|x)={}'.format(l, p) for l, p in zip(products, P)]
	# print("Posteriors: {} (n={})".format(PL, n))
	# print("p(None|x)={}".format(pNone))

	opt = F1Optimizer.maximize_expectation(P, pNone)

	if opt[1]:
		best_prediction = [None]
		best_k_proba = [pNone]
	else:
		best_prediction = []
		best_k_proba = []

	best_prediction += (products[:opt[0]])
	best_k_proba += list(P[:opt[0]])

	f1_max = opt[2]
	if f1_max < WARNING_F1:
		print 'warning f1_max {}, user_id: {}, best_prediction; {}'.format(f1_max, user_id, best_prediction)
	# 	print ['pid:{}-proba:{:.5f}'.format(l, p) for l, p in zip(products, P)]
	# print("Prediction {} yields best E[F1] of {}\n".format(best_prediction, f1_max))
	print '[lh] before len： {}, after len: {}'.format(len())
	return best_prediction, np.round(best_k_proba, 5)


def get_best_prediction_group_submit(df):
	pNone = (1.0 - df['proba']).prod()
	# save some compute
	df = df[df['proba'] > 0.01]
	df = df.sort_values('proba', ascending=False).reset_index().head(80)
	best_prediction = get_best_prediction_submit(df[PID], df['proba'], pNone)
	return pd.Series([best_prediction], index=['products'])


def get_best_prediction_submit(products, P, pNone=None):
	products = list(products)
	if pNone is None:
		pNone = (1.0 - P).prod()

	opt = F1Optimizer.maximize_expectation(P, pNone)

	if opt[1]:
		best_prediction = [None]
		# best_k_proba = [pNone]
	else:
		best_prediction = []
		# best_k_proba = []

	best_prediction += (products[:opt[0]])
	return best_prediction


def save_plot(P, filename='expected_f1.png'):
	E_F1 = pd.DataFrame(F1Optimizer.get_expectations(P).T, columns=["/w None", "/wo None"])
	best_k, _, max_f1 = F1Optimizer.maximize_expectation(P)

	plt.style.use('ggplot')
	plt.figure()
	E_F1.plot()
	plt.title('Expected F1-Score for \n {}'.format("P = [{}]".format(",".join(map(str, P)))), fontsize=12)
	plt.xlabel('k')
	plt.xticks(np.arange(0, len(P) + 1, 1.0))
	plt.ylabel('E[F1(P,k)]')
	plt.plot([best_k], [max_f1], 'o', color='#000000', markersize=4)
	plt.annotate('max E[F1(P,k)] = E[F1(P,{})] = {:.5f}'.format(best_k, max_f1), xy=(best_k, max_f1),
							 xytext=(best_k, max_f1 * 0.8), arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=7),
							 horizontalalignment='center', verticalalignment='top')
	plt.gcf().savefig(filename)


def timeit(P):
		s = datetime.now()
		F1Optimizer.maximize_expectation(P)
		e = datetime.now()
		return (e-s).microseconds / 1E6


def benchmark(n=100, filename='runtimes.png'):
	results = pd.DataFrame(index=np.arange(1,n+1))
	results['runtimes'] = 0

	for i in range(1,n+1):
			runtimes = []
			for j in range(5):
					runtimes.append(timeit(np.sort(np.random.rand(i))[::-1]))
			results.iloc[i-1] = np.mean(runtimes)

	x = results.index
	y = results.runtimes
	results['quadratic fit'] = np.poly1d(np.polyfit(x, y, deg=2))(x)

	plt.style.use('ggplot')
	plt.figure()
	results.plot()
	plt.title('Expectation Maximization Runtimes', fontsize=12)
	plt.xlabel('n = |P|')
	plt.ylabel('time in seconds')
	plt.gcf().savefig(filename)


if __name__ == '__main__':
		pass
		# print_best_prediction([0.7, 0.6, 0.05])
		# df = pd.DataFrame({UID: [1, 1, 1], PID: [1, 2, 3], 'proba': [0.7, 0.6, 0.05]})
		# print get_best_prediction_group(df)

		# print_best_prediction([0.3, 0.2], 0.57)

		# FIXME 典型案例
		# print_best_prediction([0.5, 0.2, 0.2, 0.2])
		# print_best_prediction([0.6, 0.4], 0.6)

		# print_best_prediction([0.913703, 0.905528, 0.781715, 0.732509, 0.433245, 0.288590])

		# print_best_prediction([0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.04])
		# print_best_prediction([0.5, 0.4, 0.3])
		# print_best_prediction([0.6, 0.5, 0.3, 0.2, 0.2, 0.2, 0.2])

		# print_best_prediction([0.3, 0.2, 0.2])
		# print_best_prediction([0.3, 0.2], 0.57)
		# print_best_prediction([0.9, 0.6])
		# print_best_prediction([0.5, 0.4, 0.3, 0.35, 0.33, 0.31, 0.29, 0.27, 0.25, 0.20, 0.15, 0.10])
		# print_best_prediction([0.5, 0.4, 0.3, 0.35, 0.33, 0.31, 0.29, 0.27, 0.25, 0.20, 0.15, 0.10], 0.2)
		#
		# save_plot([0.45, 0.35, 0.31, 0.29, 0.27, 0.25, 0.22, 0.20, 0.17, 0.15, 0.10, 0.05, 0.02])
		# benchmark()

		# print_best_prediction([0.4019, 0.1780, 0.1315])

		print_best_prediction([0.01, 0.01, 0.01])
		# Estimate p(None | x) as (1 - p_1) * (1 - p_2) * ... * (1 - p_n)
		# Posteriors: ['p(L1|x)=0.01', 'p(L2|x)=0.01', 'p(L3|x)=0.01'](n=3)
		# p(None | x) = 0.970299
		# Prediction[None] yields best E[F1] of 0.970299
