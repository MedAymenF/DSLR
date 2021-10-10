#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import argparse


def		main():
	# Parse arguments
	my_parser = argparse.ArgumentParser(description="Displays a histogram answering the next question: Which Hogwarts course has a homogeneous score distribution between all four houses?")
	my_parser.add_argument('--all', action='store_true', help='Display a histogram for each course.')
	args = my_parser.parse_args()

	# Read csv file
	df = pd.read_csv('dataset_train.csv', index_col='Index')
	df = df.drop(['First Name', 'Last Name', 'Birthday', 'Best Hand'], axis=1)

	# Group data by Hogwarts house
	df_by_house = dict(list(df.groupby('Hogwarts House')))
	columns = list(df.columns)
	columns.pop(columns.index('Hogwarts House'))
	houses = list(df_by_house.keys())

	if args.all:
		# Plot a histogram for each course
		plt.figure(figsize=(25, 13))
		for index, column in enumerate(columns):
			axe = plt.subplot(4, 4, index + 1)
			column_by_house = [df_by_house[house][column] for house in houses]
			plt.hist(column_by_house, label=houses, bins=20)
			plt.title(column)
			axe.set_ylabel('Frequency')
			plt.legend(loc='upper left')
	else:
		column_by_house = [df_by_house[house]['Care of Magical Creatures'] for house in houses]
		plt.hist(column_by_house, label=houses)
		plt.title('Care of Magical Creatures')
		plt.legend()
	plt.show()


if __name__ == '__main__':
	main()
