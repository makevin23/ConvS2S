import numpy as np

def dropout(X, rate):
    binary_value = np.random.rand(*X.shape) < rate
    res = np.multiply(X, binary_value)
    res /= rate  # this line is called inverted dropout technique
    print(res)
    return res

weights = np.ones([1,5])
dropout(weights, 1)