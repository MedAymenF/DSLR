#!/usr/bin/env python3
import numpy as np
import pandas as pd


def		main():
	# Read dataset_truth.csv
	try:
		truth = pd.read_csv('dataset_truth.csv', index_col=0, squeeze=True).to_numpy()
	except Exception as e:
		print(f"Can't open dataset_truth.csv or it's not a valid csv file.")
		print(e)
		exit(1)

	# Read houses.csv
	try:
		predictions = pd.read_csv('houses.csv', index_col=0, squeeze=True).to_numpy()
		assert len(predictions) == len(truth)
	except Exception as e:
		print(f"Can't open houses.csv or it's not a valid csv file.")
		print(e)
		exit(1)

	# Calculate accuracy
	print(f'Accuracy on test set = {np.mean(predictions == truth) * 100}%')


if __name__ == '__main__':
	main()
