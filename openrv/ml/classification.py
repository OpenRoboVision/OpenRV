import sklearn.svm as _svm_super
import sklearn.neighbors as _knn_super
import sklearn.naive_bayes as _nb_super

import numpy as np
from openrv.ml import MlObject


class _base:

	def train(self, dataset, verbose=0, max_iter=10000):
		self.classifier.verbose = verbose
		self.classifier.max_iter = max_iter

		y, X = list(zip(*dataset.to_list()))

		if len(np.array(X).shape) == 1:
			X = np.array(X).reshape(len(X), 1).tolist()
		X = list(X)
		y = list(y)
		self.classifier.fit(X, y)

	def test_xy(self, X, y):
		if type(X) == MlObject:
			X = self._prepare(X.data)
		else:
			X = self._prepare(X)
		return 0

	def test(self, dataset):
		X = self._prepare(dataset.datas)
		y = dataset.labels
		return self.classifier.score(X, y)

	def predict(self, X):
		X = self._prepare(X)
		try:
			return self.classifier.predict(X)
		except ValueError:
			return self.classifier.predict(X.reshape(1, -1))

	def set_param(self, param_name, value):
		setattr(self.classifier, param_name, value)


	def _prepare(self, X):
		if type(X) == MlObject:
			X = X.data
		X = np.array(X)
		return X


class SVM(_base):

	def __init__(self, method, **params):
		if method == 'SVC':
			self.classifier = _svm_super.SVC()
		elif method == 'NuSVC':
			self.classifier = _svm_super.NuSVC()
		elif method == 'Linear':
			self.classifier = _svm_super.LinearSVC()
		else:
			raise ValueError(f'Unsuppoprted method {method}')

		params = dict(params)
		if not ('gamma' in params.keys()):
			params['gamma'] = 'auto'

		for param, val in params.items():
			self.set_param(str(param), val)


class KNN(_base):

	def __init__(self, N, **params):
		self.classifier = _knn_super.KNeighborsClassifier(n_neighbors=N)

		params = dict(params)
		if not ('gamma' in params.keys()):
			params['gamma'] = 'auto'

		for param, val in params.items():
			self.set_param(str(param), val)


class GaussianNB(_base):

	def __init__(self, **params):
		self.classifier = _nb_super.GaussianNB()

		params = dict(params)

		for param, val in params.items():
			self.set_param(str(param), val)	


class BernoulliNB(_base):

	def __init__(self, **params):
		self.classifier = _nb_super.BernoulliNB()

		params = dict(params)

		for param, val in params.items():
			self.set_param(str(param), val)


class ComplementNB(_base):

	def __init__(self, **params):
		self.classifier = _nb_super.ComplementNB()

		params = dict(params)

		for param, val in params.items():
			self.set_param(str(param), val)	