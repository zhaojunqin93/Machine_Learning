import numpy as np


def sigmoid(x):

    s = 1 / (1 + np.exp(-x))

    return s

def sigmoid_derivative(x):

    s = sigmoid(x)
    ds = s * (1 - s)

    return ds

def image2vector(image):

    v = image.reshape(image.shape[0] * image.shape[1] * image.shape[2], 1)

    return v

def normalizeRows(x):

    x_norm = np.linalg.norm(x, axis = 0, keepdims = True)
    x = x / x_norm

    return x
#
# x = np.array([
#     [0, 3, 4],
#     [1, 6, 4]])
# print("normalizeRows(x) = " + str(normalizeRows(x)))


def softmax(x):

    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis=1, keepdims = True)
    s = x_exp / x_sum

    return s
# x = np.array([
#     [9, 2, 5, 0, 0],
#     [7, 5, 0, 0 ,0]])
# print("softmax(x) = " + str(softmax(x)))

def loss1(y_hat, y):

    loss = np.sum(np.abs(y- y_hat))

    return loss

def loss2(y_hat, y):

    loss = np.dot((y - y_hat), (y - y_hat).T)

    return loss

yhat = np.array([.9, 0.2, 0.1, .4, .9])
y = np.array([1, 0, 0, 1, 1])
print("L2 = " + str(loss2(yhat,y)))
# image = np.array([[[ 0.67826139,  0.29380381],
#         [ 0.90714982,  0.52835647],
#         [ 0.4215251 ,  0.45017551]],
#
#        [[ 0.92814219,  0.96677647],
#         [ 0.85304703,  0.52351845],
#         [ 0.19981397,  0.27417313]],
#
#        [[ 0.60659855,  0.00533165],
#         [ 0.10820313,  0.49978937],
#         [ 0.34144279,  0.94630077]]])
# print(image.shape)
# print("image2vector:", image2vector(image))
# x = np.random.randint(low=1, high=5, size=(2,3))
# print(x)
# print("sigmoid_derivative:", sigmoid_derivative(x))