import numpy as np
import random

def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))


class FullyConnectedNN:
    
    def __init__(self, sizes):
        """Initialize fully connected NN using the list size"""
        self.sizes = sizes
        self.W = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        self.B = [np.random.randn(y, 1) for y in sizes[1:]]

    def feedforward(self, x):
        for w, b in zip(self.W, self.B):
            x = sigmoid(np.dot(w, x) + b)
        return x

    def cost(self, output, y):
        return 0.5*(output - y)*(output - y)

    def cost_derivative(self, output, y):
        return (output-y) 

    def evaluate(self, test_data):
        test_results = []
        for i in range(0,len(test_data)):
            test_results.append(
                (np.argmax(self.feedforward(test_data[i][0])), 
                 test_data[i][1]))
        return sum(int(x == y) for (x, y) in test_results) \
                     / float(len(test_data))

    def backprop(self, trn_sample):
        # Return nabla_b and nabla_w, the cost gradients
        # with respect to B and W
        nabla_b = [np.zeros(b.shape) for b in self.B]
        nabla_w = [np.zeros(w.shape) for w in self.W]

        # Feedforward
        activation = trn_sample[0]
        activations = [trn_sample[0]]
        zs = [] # list of layer zs
        for b, w in zip(self.B, self.W):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # Backward pass
        delta = self.cost_derivative(activations[-1], trn_sample[1]) * \
             sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in xrange(2, len(self.sizes)):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.W[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())

        return (nabla_b, nabla_w)
    
    def train(self, trn_data, nEpoch, mini_size, 
              eta, test_data=None):
        """
        Uses mini-batch stochastic gradient descent and backpropogation.
        trn_data should be a list of tuples (x, y). In each such tuple
        x is an ndarray of input values and y is a label ndarray.
        For MNIST example, load data using mnist_loader.py and run the net as follows:
        > import mnist_loader
        > trn_data, val_data, test_data = mnist_loader.load_data_wrapper()
        > nn = FullyConnectedNN(sizes)
        > nn.train(trn_data, nEpoch,...)
        """
        num_sample = len(trn_data)
        for e in xrange(nEpoch):    
            random.shuffle(trn_data)
            i = 0
            while i < num_sample:
                # Update on minibatch
                nabla_b = [np.zeros(b.shape) for b in self.B]
                nabla_w = [np.zeros(w.shape) for w in self.W]
                for j in xrange(0,mini_size):
                    dnabla_b, dnabla_w = self.backprop(trn_data[i])
                    nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, dnabla_b)]
                    nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, dnabla_w)]
                    i += 1
                self.B = [b - (eta/float(mini_size))*nb for b, nb in zip(self.B, nabla_b)]
                self.W = [w - (eta/float(mini_size))*nw for w, nw in zip(self.W, nabla_w)]

            # Evaluate on test set
            if test_data:
                if nEpoch > 20:
                    if e % (nEpoch/10) == 0:
                        print "Epoch", e, "Accuracy: ", \
                            self.evaluate(test_data)
                else:
                    print "Epoch", e, "Accuracy: ", \
                        self.evaluate(test_data)

