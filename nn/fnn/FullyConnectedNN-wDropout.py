
# coding: utf-8

# In[1]:


import numpy as np
import csv
import os
import random
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# In[21]:


a = np.ones(shape=[3,2])
b = np.random.rand(3,2)
print a
print b

c = b>0.5
print c
m = np.multiply(b,c)
print m


# In[4]:


sizes = [10,5,2,2]
W = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
B = [np.random.randn(y, 1) for y in sizes[1:]]

Wrands = [np.random.rand(y,x) for x, y in zip(sizes[:-1], sizes[1:])]
Brands = [np.random.rand(y, 1) for y in sizes[1:]]

Wdrops = [w>0.4 for w in Wrands]
Bdrops = [b>0.4 for b in Brands]

Wtmp = np.multiply(W,Wdrops)
Btmp = np.multiply(B,Bdrops)


# Wones = [np.ones(shape=[y,x]) for x, y in zip(sizes[:-1], sizes[1:])]
# Bones = [np.ones(shape=[y,1]) for y in sizes[1:]]

print "W",W
print "B",B
print "Wdrops", Wdrops
print "Bdrops", Bdrops
print "Wtmp", Wtmp
print "Btmp", Btmp


# In[ ]:


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
        
        # create dropout filter, initially all True
        Wrands = [np.random.rand(y,x) for x, y in zip(sizes[:-1], sizes[1:])]
        Brands = [np.random.rand(y, 1) for y in sizes[1:]]
        self.Wdrops = [wr>0. for wr in Wrands]
        self.Bdrops = [br>0. for br in Brands]


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
        return sum(int(x == y) for (x, y) in test_results)                     / float(len(test_data))

    def backprop(self, trn_sample, dropProb):
        # Return nabla_b and nabla_w, the cost gradients
        # with respect to B and W
        nabla_b = [np.zeros(b.shape) for b in self.B]
        nabla_w = [np.zeros(w.shape) for w in self.W]
        
        W_ = np.multiply(self.W,self.Wdrops)
        B_ = np.multiply(self.B,self.Bdrops)

        # Feedforward
        activation = trn_sample[0]
        activations = [trn_sample[0]]
#         activation = trn_sample
#         activations = [trn_sample]
        zs = [] # list of layer zs
    
        for b_, w_ in zip(B_, W_):
            z = np.dot(w_, activation) + b_
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
                

        # Backward pass
        delta = self.cost_derivative(activations[-1], trn_sample[1]) *             sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in xrange(2, len(self.sizes)):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(W_[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())

        return (nabla_b, nabla_w)
    
    def train(self, trn_data, nEpoch, mini_size, 
              eta, dropProb, test_data=None):
        num_sample = len(trn_data)
        for e in xrange(nEpoch):    
            random.shuffle(trn_data)
            i = 0
            while i < num_sample:
                # update on minibatch
                nabla_b = [np.zeros(b.shape) for b in self.B]
                nabla_w = [np.zeros(w.shape) for w in self.W]
                
                # randomize dropout filter
                Wrands = [np.random.rand(y,x) for x, y in zip(sizes[:-1], sizes[1:])]
                Brands = [np.random.rand(y, 1) for y in sizes[1:]]
                self.Wdrops = [wr>dropProb for wr in Wrands]
                self.Bdrops = [br>dropProb for br in Brands]

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
                        print "Epoch", e, "Accuracy: ",                             self.evaluate(test_data)
                else:
                    print "Epoch", e, "Accuracy: ",                         self.evaluate(test_data)

