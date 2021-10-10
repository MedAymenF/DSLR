import numpy as np
from tqdm import tqdm


def     sigmoid(z):
    return 1. / (1 + np.exp(-z))


def     cost_gradient(theta, x, y):
	predictions = sigmoid(x @ theta)
	return x.transpose() @ (predictions - y) / len(y)


def		train(initial_theta, cost_gradient, x, y, n_epochs, learning_rate):
	theta = initial_theta
	for _ in tqdm(range(n_epochs)):
		theta -= learning_rate * cost_gradient(theta, x, y)
	return theta


def		stochastic_train(initial_theta, x, y, n_epochs, learning_rate):
	np.seterr(all="ignore")
	theta = initial_theta
	for _ in tqdm(range(n_epochs)):
		for i in range(x.shape[0]):
			theta -= learning_rate * (x[i, :] * (sigmoid(x[i, :] @ theta) - y[i]))
	return theta


def		calculate_predictions(x, weights, classes):
	predictions_table = sigmoid(x @ weights)
	predictions = predictions_table.argmax(axis=1)
	predictions = classes[predictions]
	return predictions
