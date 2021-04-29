import pandas as pd
from pyspark.ml.regression import LinearRegression, GeneralizedLinearRegression, IsotonicRegression
from pyspark.ml.classification import LogisticRegression, NaiveBayes, LinearSVC
from sklearn import linear_model
from sklearn import svm
from sklearn.model_selection import train_test_split

# author Dennis Biber
# Date: 28/4/2021

class SciKitSparkLib(object):

	def __init__(self, dataframe=None, csvPath=False, filepath=None, test_size=0.2):
		if csvPath:
			self.df = pd.read_csv(filepath, header=0)
		elif dataset != None:
			self.df = dataframe
		else:
			raise "You did not provide any data! Specify a dataset or csvPath."
		self.test_size = test_size

	def scalarTTS(self):
		self.train, self.test = train_test_split(self.df, test_size=self.test_size)

	def binomialTTS(self, key):
		X = self.df.pop(key)
		y = self.df
		self.Xtrain, self.Xtest, self.Ytrain, self.Ytest = train_test_split(X.index. y, test_size=self.test_size)

	def scalarSciKitLinearRegression(self):
		regr = linear_model.LinearRegression()
		regr.fit(self.train)
		return regr

	def scalarSparkLinearRegression(self):
		regr = LinearRegression()
		model = regr.fit(self.train)
		return model

	def scalarSparkGLR(self):
		regr = GeneralizedLinearRegression()
		model = regr.fit(self.train)
		return model

	def scalarSparkIsoRegression(self):
		regr = IsotonicRegression()
		model = regr.fit(self.train)
		return model

	def scalarSparkLinearSVM(self):
		model = LinearSVC()
		model = model.fit(self.train)
		return model

	def scalarSparkLogisticRegression(self):
		regr = LogisticRegression()
		model = regr.fit(self.train)
		return model

	def scalarSparkNaiveBayes(self):
		model = NaiveBayes()
		model = model.fit(self.train)
		return model

	def binomialSciKitLinearRegression(self):
		regr = linear_model.LinearRegression()
		regr.fit(self.Xtrain, self.Ytrain)
		return regr

	def binomialSparkLinearRegression(self):
		regr = LinearRegression()
		model = regr.fit(self.Xtrain, self.Ytrain)
		return model

	def binomialSparkGLF(self):
		regr = GeneralizedLinearRegression()
		model = regr.fit(self.Xtrain, self.Ytrain)
		return model

	def binomialSparkIsoRegression(self):
		regr = IsotonicRegression()
		model = regr.fit(self.Xtrain, self.Ytrain)
		return model

	def binomialSparkLinearSVM(self):
		model = LinearSVC()
		model = model.fit(self.Xtrain, self.Ytrain)
		return model

	def binomialSciKitLogisticRegression(self):
		regr = linear_model.LogisticRegression()
		regr.fit(self.Xtrain, self.Ytrain)
		return regr

	def binomialSparkLogisticRegression(self):
		regr = LogisticRegression()
		model = regr.fit(self.Xtrain, self.Ytrain)
		return model

	def binomialSparkNaiveBayes(self):
		model = NaiveBayes()
		model = model.fit(self.Xtrain, self.Ytrain)
		return model

	def binomialSciKitLinearSVC(self):
		model = svm.LinearSVC(kernel='linear')
		model.fit(self.Xtrain, self.Ytrain)
		return model

	def binomialSciKitSVC(self):
		model = svm.SVC()
		model.fit(self.Xtrain, self.Ytrain)
		return model

	def binomialSparkLinearSVM(self):
		model = LinearSVC()
		model = model.fit(self.Xtrain, self.Ytrain)
		return model

	def interceptSciKit(self, model):
		return model._intercept

	def slopeSciKit(self, model):
		return model.coef_

	def interceptSpark(self, model):
		return model.intercept

	def slopeSpark(self, model):
		return model.coefficients

	def sparkRMSE(self, model):
		return model.summary.rootMeanSquaredError

	def predictScalar(self, model):
		return model.transform(self.test)

	def predictBinomial(self, model):
		return model.transofrm(self.Xtest, self.Ytest)
