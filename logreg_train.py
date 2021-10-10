#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
from gradient_descent import train, stochastic_train, cost_gradient, calculate_predictions


def		main():
	# Parse arguments
	my_parser = argparse.ArgumentParser(description='Train a logistic regression model.')
	my_parser.add_argument('Path', metavar='path', type=str, help='The path to the csv file containing the training data.')
	my_parser.add_argument('-n', '-Number of epochs', type=int, default=40000)
	my_parser.add_argument('-l', '-Learning rate', type=float, default=0.0001)
	my_parser.add_argument('-s', action='store_true', help='Use stochastic gradient descent for training.')
	args = my_parser.parse_args()
	if (args.n < 1):
		print('The number of epochs must be larger than zero.')
		exit(1)
	if (args.l <= 0):
		print('The learning rate must be larger than zero.')
		exit(1)

	# Read training data
	try:
		df = pd.read_csv(args.Path)
	except Exception as e:
		print(f"Can't open {args.Path} or it's not a valid csv file.")
		print(e)
		exit(1)

	# Select features to use in the logistic regression
	selected_features = ['Astronomy', 'Herbology', 'Divination', 'Muggle Studies', 'Ancient Runes', 'History of Magic', 'Transfiguration', 'Flying']

	# Save each feature's mean to a csv file to fill na values in the test dataset
	means = df[selected_features].mean()
	means.to_csv('means.csv')

	# Fill na values with mean per class
	df[selected_features] = df.groupby('Hogwarts House')[selected_features].transform(lambda x: x.fillna(x.mean()))

	# Select labels
	labels = df['Hogwarts House']
	labels = labels.to_numpy()
	hogwart_houses = np.unique(labels)

	# Create feature matrix
	df = df[selected_features]
	x = df.to_numpy()
	n_theta = x.shape[1] + 1
	features = np.ones((x.shape[0], n_theta))
	features[:, 1:] = x

	# Create the numpy array containing the weights for each classifier
	classifiers = np.zeros(shape=(len(hogwart_houses), n_theta))

	# Training
	for i, house in enumerate(hogwart_houses):
		print(f'Training classifier for {house}')
		initial_theta = np.zeros(n_theta)
		y = (labels == house).astype(int)
		if (args.s):
			theta = stochastic_train(initial_theta, features, y, args.n, args.l)
		else:
			theta = train(initial_theta, cost_gradient, features, y, args.n, args.l)
		classifiers[i, :] = theta

	# Calculate accuracy
	predictions = calculate_predictions(features, classifiers.transpose(), hogwart_houses)
	print(f'\nAccuracy on training set = {np.mean(predictions == labels) * 100}%')

	# Save models' weights to a csv file
	weights = pd.DataFrame(classifiers.transpose(), columns=hogwart_houses)
	weights.to_csv('weights.csv', index_label='Index')
	print('\nWeights were saved in weights.csv')


if __name__ == '__main__':
	main()
