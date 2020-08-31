import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from ml_models.logistic_regression.logistic_regression import LogisticRegression

import warnings
warnings.filterwarnings('error')

def load_dataset(path):
	dataset = pd.read_csv(path)

	return dataset

def split_data(dataset):
	data = dataset.loc[:, dataset.columns != 'target'].to_numpy()
	labels = dataset['target'].to_numpy()

	x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=123)

	return x_train, x_test, y_train, y_test

def main():
	path = 'data/heart_processed.csv'
	dataset = load_dataset(path)

	x_train, x_test, y_train, y_test = split_data(dataset)

	lg = LogisticRegression(100, 0.1, verbose=True)
	lg.fit(x_train, y_train)

if __name__ == '__main__':
	main()
