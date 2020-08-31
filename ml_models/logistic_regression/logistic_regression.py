import pandas as pd
import numpy as np

from typing import Union
import warnings
warnings.filterwarnings('error')

class LogisticRegression:
	def __init__(self, num_iterations: int, alpha: float, verbose: bool = False):
		'''Initialize LogisticRegression'''
		self.num_iterations = num_iterations
		self.alpha = alpha
		self.verbose = verbose
		self.epsilon = 1e-5

		self.predictions = None
		self.score = None

	def _sigmoid(self, z: np.complex128) -> Union[np.complex128, np.ndarray]:
		'''Sigmoid function to do non-linearity transformation

		Args:
			z (float): calculation of weights and data

		Returns:
			sigmoid (float or np.array[float]): non-linearity applied to calculation
		'''
		try:
			sigmoid = 1 / (1 + np.exp(-z) + self.epsilon)
		except Warning:
			sigmoid = self.epsilon

		return sigmoid

	def _add_intercept(self, X: np.ndarray) -> np.ndarray:
		'''Adds additional intercept to dataset

		Args:
			X (np.ndarray): dataset (train or test)

		Returns:
			X_intercept (np.ndarray): dataset with additional intercept
		'''
		intercept = np.ones((X.shape[0]), 1)
		X_intercept = np.concatenate((intercept, X), axis=1)

		return X_intercept

	def _loss_function(self, h: np.ndarray, y: np.ndarray) -> float:
		'''Utilizing logistic loss. Calculating in parts, equation is:
		loss(...) = 1/m * [(-y * log(h)) - ((1-y) * log(1-h)]

		Args:
			h (np.array): training dataset
			y_train (np.array): labels dataset

		return:
			loss (float): cost value of the prediction
		'''
		loss = (-y * np.log(h)) - ((1 - y) * np.log(1 - h))
		loss = loss.mean()

		return loss

	def _gradient_descent(self, x_train: np.ndarray, y_train: np.ndarray, weights: np.ndarray, num_iteration: int):
		'''Gradient descent to find minima of function

		Args:
			x_train (np.ndarray): training dataset
			y_train (np.ndarray): labels dataset
			weights (np.ndarray): weights vector
			num_iteration (int): number of iterations to optimize
		'''
		num_data = y_train.size

		z = np.dot(x_train, weights)
		h = self._sigmoid(z)

		loss = self._loss_function(h, y_train)
		gd = np.dot(x_train.T, (h - y_train)) / num_data

		self.weights = self.weights - (self.alpha * gd)

		if self.verbose and num_iteration % 10 == 0:
			self._verbose(num_iteration, loss)

	def _verbose(self, num_iteration: int, loss: float):
		'''Displays loss over time (iterations)

		Args:
			num_iteration (int): current iteration
			loss (float): current loss
		'''
		v_message = "Iteration: {0} - Loss: {1}"
		print(v_message.format(num_iteration, loss))

	def fit(self, x_train: np.ndarray, y_train: np.ndarray):
		'''Fit the training data into the machine learning model

		Args:
			x_train (np.ndarray): training dataset
			y_train (np.ndarray): labels dataset
		'''
		self.weights = np.zeros(x_train.shape[1])

		for num_iteration in range(self.num_iterations):
			self._gradient_descent(x_train, y_train, self.weights, num_iteration)

	def predict(self, x_test: np.ndarray, threshold: float = 0.5) -> np.ndarray:
		'''Prediction function to guess new data

		Args:
			x_test (np.ndarray): test dataset OR production dataset
			threshold (float): threshold to determine is class 1/0

		Returns:
			predictions (np.ndarray): predictions from ML algorithm
		'''
		predictions = self._predict_proba(x_test) >= threshold
		return predictions

	def _predict_proba(self, x_test: Union[np.ndarray, float]) -> Union[float, np.ndarray]:
		'''Calculating the sigmoid of the data

		Args:
			x_test (np.ndarray): test dataset OR production dataset

		Returns:
			res_sigmoid (float or np.ndarray):
		'''
		x_test = self._add_intercept(x_test)
		res_sigmoid = self._sigmoid(np.dot(x_test, self.weights))

		return res_sigmoid
