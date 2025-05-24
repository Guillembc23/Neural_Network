import numpy as np
import matplotlib.pyplot as plt
import random

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_der(x):
    return sigmoid(x) * (1 - sigmoid(x))

def relu(x):
    return np.maximum(0, x)

def relu_der(x):
    return (x > 0).astype(float)

def tanh(x):
    return np.tanh(x)

def tanh_der(x):
    return 1 - np.tanh(x)**2

class Layer:
    def __init__(self, input_size, output_size, activation, activation_der):
        self.weights = np.random.randn(input_size, output_size)  #weights[i,j]
        self.biases = np.zeros((1, output_size))
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        self.activation_der = activation_der

    def forget(self):
        i = random.randint(0, self.input_size - 1)
        j = random.randint(0, self.output_size - 1)
        self.weights[i,j] = 0


    def forward(self, input_data):
        self.input = input_data
        self.z = np.dot(input_data, self.weights) + self.biases
        self.output = self.activation(self.z)
        return self.output

    def backward(self, output_error, lr):
        d_activation = self.activation_der(self.z) #d_activation[j]
        delta = output_error * d_activation #delta[k,j]

        dw = np.dot(self.input.T, delta) #dw[i,j] = input[k,i]*delta[k,j] (automatically sum over all k)
        db = np.sum(delta, axis=0, keepdims=True)

        prev_error = np.dot(delta, self.weights.T) #prev_error[k,i] = sum delta[k,j]*w[i,j]

        self.weights -= lr * dw
        self.biases -= lr * db

        return prev_error
    
class Network:
    def __init__(self,layers):
        self.layers = layers
    
    def forward(self,X):
        for layer in self.layers:
            X = layer.forward(X)
        return X
    
    def backward (self, y_pred, y_true, lr):
        error = 2*(y_pred - y_true)/ y_true.shape[0]
        for layer in reversed(self.layers):
            error = layer.backward(error, lr = lr)

    def forget(self):
        for layer in self.layers:
            layer.forget()

    def train (self, X, y, epochs, lr = 0.001):
        loss = []
        for epoch in range(epochs):
            output = self.forward(X)
            loss.append(np.mean((output - y) ** 2))
            self.backward(output,y, lr)
            self.forget()

        plt.plot(loss)

        plt.title('Loss vs Training')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()
            

    def predict(self,X):
        return self.forward(X)



