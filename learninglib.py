from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model
from pyspark.ml.regression import LinearRegression
import pandas as pd

# author Dennis Biber
# Date: 28/4/2021

class LearningLib(object):

	def __init__(self, dataset=None, csvPath=False, filepath=None, test_size=0.2):
		if csvPath:
			self.df = pd.read_csv(filepath, header=0)
		elif dataset != None:
			self.df = dataset
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

	def binomialSciKitLinearRegression(self):
		regr = linear_model.LinearRegression()
		regr.fit(self.Xtrain, self.Ytrain)
		return regr

	def scalarSparkLinearRegression(self):
		regr = LinearRegression()
		model = regr.fit(self.train)
		return model

	def binomialSparkLinearRegression(self):
		regr = LinearRegression()
		model = regr.fit(self.Xtrain, self.Ytrain)
		return model

	def binomialLogisticRegression(self):
		regr = linear_model.LogisticRegression()
		regr.fit(self.Xtrain, self.Ytrain)
		return regr

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

	def predictScalarSpark(self, model):
		return model.transform(self.test)

	def predictBinomialSpark(self, model):
		return model.transofrm(self.Xtest, self.Ytest)
