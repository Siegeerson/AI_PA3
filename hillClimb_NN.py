import numpy as np
import scipy.special
import torch
import torch.nn as torchNN

# A few useful resources:
#
# NumPy Tutorial:
#    https://docs.scipy.org/doc/numpy/user/quickstart.html
#
# Backpropogation Calculus by 3Blue1Brown:
#    https://www.youtube.com/watch?v=tIeHLnjs5U8
#
# Make Your Own Neural Network:
#   https://www.amazon.com/Make-Your-Own-Neural-Network-ebook/dp/B01EER4Z4G
#

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1.0 - x)

class hillClimb:
    def __init__(self, n_input, n_hidden, n_output):
        # Set number of nodes in each input, hidden, output layer
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output

        # We link the weight matrices below.
        # Weights between input and hidden layer = weights_ih
        # Weights between hidden and output layer = weights_ho
        # Weights inside the arrays are w_i_j, where the link is from node i to
        # node j in the next layer

        #   w11 w21
        #   w12 w22 etc...
        #
        # Weights are sampled from a normal probability distribution centered at
        # zero with a standard deviation related to the number of incoming links
        # into a node: 1/√(number of incoming links).
        self.weights_ih = np.zeros((self.n_hidden, self.n_input))
        self.weights_ho = np.zeros((self.n_output, self.n_hidden))


    def getMutation(self):
        mutih = np.random.uniform(-.01,.01,(self.n_hidden, self.n_input))
        mutho = np.random.uniform(-.01,.01,(self.n_output, self.n_hidden))
        newNN = hillClimb(self.n_input,self.n_hidden,self.n_output)
        newNN.weights_ih = self.weights_ih+mutih
        newNN.weights_ho = self.weights_ho+mutho
        return newNN


    # Query the neural network with simple feed-forward.
    def query(self, inputs_list):
        # Convert inputs list to 2d array.
        inputs = np.array(inputs_list, ndmin=2).T
        # Calculate signals into hidden layer.
        hidden_inputs = np.dot(self.weights_ih, inputs)
        # Calculate the signals emerging from hidden layer.
        hidden_outputs = sigmoid(hidden_inputs)
        # Calculate signals into final output layer.
        final_inputs = np.dot(self.weights_ho, hidden_outputs)
        # Calculate the signals emerging from final output layer.
        final_outputs = sigmoid(final_inputs)
        return final_outputs

# input_nodes = 2
# hidden_nodes = 6
# output_nodes = 1
# learning_rate = 0.3
#
# Creates an instance of the scratch neural network.
# Here we teach it how to produce correct "XOR" output.
# n = ScratchNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
# X = [[0,0],
#      [0,1],
#      [1,0],
#      [1,1]]
# y = [[0],
#      [1],
#      [1],
#      [0]]
#
# print('Before:', n.query(X))
# for _ in range(5000):
#     n.train(X, y)
# print('After', n.query(X))

# Before: [[0.60018041 0.60921318 0.74879427 0.72999071]]
# After [[0.02426062 0.98082423 0.97728005 0.01916943]]
