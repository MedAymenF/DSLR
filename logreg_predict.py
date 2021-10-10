#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
from gradient_descent import calculate_predictions


def		main():
	# Parse arguments
	my_parser = argparse.ArgumentParser(description='Predict which Hogwart house a student belongs to based on their scores.')
	my_parser.add_argument('data_path', type=str, help='The path to the csv file containing the test data.')
	my_parser.add_argument('weights_path', type=str, help='The path to the csv file containing the weights.')
	args = my_parser.parse_args()

	# Read test data
	try:
		df = pd.read_csv(args.data_path, index_col='Index')
	except Exception as e:
		print(f"Can't open {args.data_path} or it's not a valid csv file.")
		print(e)
		exit(1)

	# Fill na values with (training dataset) mean
	try:
		means = pd.read_csv('means.csv', squeeze=True, index_col=0)
	except Exception as e:
		print(f"Can't open {args.data_path} or it's not a valid csv file.")
		print(e)
		exit(1)
	df = df.fillna(means)

	# Select features to use in the logistic regression
	try:
		df = df[means.index]
	except Exception as e:
		print(f'{args.data_path} is not a valid test dataset.')
		print(e)
		exit(1)

	# Create feature matrix
	x = df.to_numpy()
	n_theta = x.shape[1] + 1
	features = np.ones((x.shape[0], n_theta))
	features[:, 1:] = x

	# Load weights
	try:
		weights = pd.read_csv(args.weights_path, index_col='Index')
		hogwart_houses = np.array(weights.columns)
		weights = weights.to_numpy()
		assert weights.shape == (n_theta, len(hogwart_houses))
	except Exception as e:
		print(f"Can't open {args.weights_path} or it's not a valid csv file.")
		print(e)
		exit(1)

	# Calculate predictions
	predictions = calculate_predictions(features, weights, hogwart_houses)

	# Save predictions to csv file
	houses = pd.DataFrame(predictions, columns=['Hogwarts House'])
	houses.to_csv('houses.csv', index_label='Index')


if __name__ == '__main__':
	main()
