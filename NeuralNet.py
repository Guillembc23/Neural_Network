import numpy as np
import matplotlib.pyplot as plt
import random
import pickle
import gzip

class Functional:
    def fun():
        raise NotImplementedError

    def forward(self,x):
        self.z = self.fun(x)
        return self.z

    def der():
        raise NotImplementedError

    def backward(self, output_error, lr):
        return output_error * self.der(self.z)
    
class Sigmoid(Functional):
    def fun(self,x):
        return 1 / (1 + np.exp(-x))

    def der(self,x):
        return self.fun(x) * (1 - self.fun(x))
    
class Relu(Functional):
    def fun(self,x):
        return np.maximum(0, x)

    def der(self,x):
        return (x > 0).astype(float)

class Tanh(Functional):
    def fun(self,x):
        return np.tanh(x)

    def der(self,x):
        return 1 - np.tanh(x)**2
    
class Softmax(Functional):
    def fun(self, x):
        exps = np.exp(x)  
        return exps / np.sum(exps, axis=1, keepdims=True)

    def der(self, x):
        return self.fun(x) * (1 - self.fun(x))

class Linear:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size)  #weights[i,j]
        self.biases = np.zeros((1, output_size))
        self.input_size = input_size
        self.output_size = output_size

    def forget(self):
        i = random.randint(0, self.input_size - 1)
        j = random.randint(0, self.output_size - 1)
        self.weights[i,j] = 0


    def forward(self, input_data):
        self.input = input_data
        self.z = np.dot(input_data, self.weights) + self.biases
        return self.z

    def backward(self, output_error, lr):
        delta = output_error  #delta[k,j]

        dw = np.dot(self.input.T, delta) #dw[i,j] = input[k,i]*delta[k,j] (automatically sum over all k)
        db = np.sum(delta, axis=0, keepdims=True)

        prev_error = np.dot(delta, self.weights.T) #prev_error[k,i] = sum delta[k,j]*w[i,j]

        self.weights -= lr * dw
        self.biases -= lr * db

        return prev_error
    
class Network: #sequential Neural Network
    def __init__(self,layers):
        if isinstance(layers, str):
            with gzip.open(layers, 'rb') as f:
                self.layers = pickle.load(f)['Layers']
        else:
            self.layers = layers
    
    def forward(self,X):
        for layer in self.layers:
            X = layer.forward(X)
        return X
    
    def backward (self, error, lr):

        for layer in reversed(self.layers):
            error = layer.backward(error, lr = lr)

        return error


    def train (self, X, y, epochs, lr = 1):
        loss = []
        for epoch in range(epochs):
            output = self.forward(X)
            curr_loss = np.mean((output - y) ** 2)
            loss.append(curr_loss)
            print(f"Epoch: {epoch}, loss: {curr_loss}")
            error = 2*(output - y)/ y.shape[0]
            self.backward(error, lr)

        plt.plot(loss)

        plt.title('Loss vs Training')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()
            

    def predict(self,X):
        return self.forward(X)
    
    def params(self):
        map = {}
        map['Architecture'] = [layer.__class__.__name__ for layer in self.layers]
        map['Layers'] = [layer for layer in self.layers]

        return map
    
    def save(self, name):
        with gzip.open(name, 'wb') as f:
            pickle.dump(self.params(), f)





