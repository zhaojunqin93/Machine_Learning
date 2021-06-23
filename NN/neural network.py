import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import  load_breast_cancer
from sklearn.model_selection import train_test_split

def initialize_parameters(layer_dims):
	L = len(layer_dims)
	parameters = {}
	for l in range(1, L):
		parameters["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.1
		parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))
	return parameters

def linear_forward(x, w, b):
	z = np.dot(w, x) + b
	return z

def relu_forward(z):
	A = np.maximum(0, z)
	return A

def sigmoid(z):
	A = 1/ (1 + np.exp(-z))
	return A

def forward_propagation(X, parameters):
	L = len(parameters) // 2
	A = X
	caches = []
	# calculate from 1 to L-1 layer
	for l in range(1, L):
		W = parameters["W" + str(l)]
		b = parameters["b" + str(l)]
		# linear forward -> activation function -> linear forward......
		z = linear_forward(A, W, b)
		caches.append((A, W, b, z))
		A = relu_forward(z)
	# calculate the Lth layer
	WL = parameters["W" + str(L)]
	bL = parameters["b" + str(L)]
	zL = linear_forward(A, WL, bL)
	caches.append((A, WL, bL, zL))
	AL = sigmoid(zL)

	return AL, caches

def compute_cost(AL, Y):
	m = Y.shape[1]
	cost = 1. / m * np.nansum(np.multiply(-np.log(AL), Y) +
							  np.multiply(-np.log(1 - AL), 1 - Y))
	cost = np.squeeze(cost)
	return cost

def relu_backpropagation(dA, Z):
	dout = np.multiply(dA, np.int64(Z > 0))
	return dout

def linear_backpropagation(dz, cache):
	A, W, b, z = cache
	dW = np.dot(dz, A.T)
	db = np.sum(dz, axis=1, keepdims=True)
	da = np.dot(W.T, dz)
	return da, dW, db

def back_propagation(AL, Y, caches):
	m = Y.shape[1]
	L = len(caches) - 1

	# caclucate the Lth layer gradients
	dz = 1. / m * (AL - Y)
	da, dWL, dbL = linear_backpropagation(dz, caches[L])
	gradients = {"dW" + str(L+1): dWL, "db" + str(L+1): dbL}

	# calculate from L-1 to 1 layer gradients
	for l in reversed(range(0, L)):
		A, W, b, z = caches[l]

		dout = relu_backpropagation(da, z)

		da, dW, db = linear_backpropagation(dout, caches[l])

		gradients["dW" + str(l+1)] = dW
		gradients["db" + str(l+1)] = db
	return gradients

def update_parameters(parameters, grads, learning_rate):
	L = len(parameters) // 2
	for l in range(L):
		parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
		parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
	return parameters

def L_layer_model(X, Y, layer_dims, learning_rate, num_iterations):
	costs = []
	print("layer_dims", layer_dims)
	parameters = initialize_parameters(layer_dims)

	for i in range(0, num_iterations):
		AL, caches = forward_propagation(X, parameters)
		cost = compute_cost(AL, Y)
		if i % 1000 == 0:
			print("Cost after iteration {}: {}".format(i, cost))
			costs.append(cost)
		grads = back_propagation(AL, Y, caches)

		parameters = update_parameters(parameters, grads, learning_rate)
	print('length of cost')
	print(len(costs))
	plt.clf()
	plt.plot(costs)
	plt.xlabel("iterations/thousand")  # 横坐标名字
	plt.ylabel("cost")  # 纵坐标名字
	plt.show()
	return parameters

def predict(x_test, y_test, parameters):
	m = y_test.shape[1]
	Y_prediction = np.zeros((1, m))
	prob, caches = forward_propagation(x_test, parameters)
	for i in range(prob.shape[1]):
		if prob[0, i] > 0.5:
			Y_prediction[0, i] = 1
		else:
			Y_prediction[0, i] = 0
	accuracy = 1- np.mean(np.abs(Y_prediction - y_test))

	return accuracy

#DNN model
def DNN(X_train, y_train, X_test, y_test, layer_dims, learning_rate= 0.001, num_iterations=30000):
	parameters = L_layer_model(X_train, y_train, layer_dims, learning_rate, num_iterations)
	accuracy = predict(X_test,y_test,parameters)
	return accuracy

if __name__ == "__main__":
	X_data, y_data = load_breast_cancer(return_X_y=True)
	X_train, X_test,y_train,y_test = train_test_split(X_data, y_data, train_size=0.8,random_state=28)
	X_train = X_train.T
	y_train = y_train.reshape(y_train.shape[0], -1).T
	X_test = X_test.T
	y_test = y_test.reshape(y_test.shape[0], -1).T
	accuracy = DNN(X_train,y_train,X_test,y_test,[X_train.shape[0],10,5,1])
	print(accuracy)