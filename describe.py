#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np


def		quantile(q, feature):
	position = (len(feature) - 1) * q / 100
	index = int(position)
	fraction = position - index
	return feature[index] + (feature[index + 1] - feature[index]) * fraction


def		main():
	# Parse arguments
	my_parser = argparse.ArgumentParser(description="Generate descriptive statistics.")
	my_parser.add_argument("File", metavar="file", type=str, help="The path to the csv file containing the data.")
	args = my_parser.parse_args()

	# Read csv file
	try:
		df = pd.read_csv(args.File)
	except Exception as e:
		print(f"Can't open {args.File}")
		print(e)
		exit(1)

	# Select numeric features
	df = df.select_dtypes(include='number')

	# Create the dataframe containing the statistics
	desc = pd.DataFrame(index=['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'], columns=df.columns, dtype='float')

	# Fill the dataframe
	for column in df.columns:
		feature = df[column].dropna()
		values = []
		# count
		count = len(feature)
		if (not count):
			values = [0] + [np.nan] * 7
			desc[column] = values
			continue
		values.append(count)
		# mean
		mean = sum(feature) / count
		values.append(mean)
		# standard deviation
		variance = sum((feature - mean)**2) / (count - 1)
		std = variance**.5
		values.append(std)
		# minimum
		feature = feature.sort_values(ignore_index=True)
		minimum = feature[0]
		values.append(minimum)
		# first quartile
		q1 = quantile(25, feature)
		values.append(q1)
		# second quartile
		q2 = quantile(50, feature)
		values.append(q2)
		# third quartile
		q3 = quantile(75, feature)
		values.append(q3)
		# maximum
		maximum = feature.iloc[-1]
		values.append(maximum)
		desc[column] = values
	print(desc)


if __name__ == '__main__':
	main()
