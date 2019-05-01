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

class ScratchNetwork:
    def __init__(self, n_input, n_hidden, n_output, learning_rate=0.3):
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
        random_init_range = pow(self.n_input, -0.5)
        self.weights_ih = np.random.normal(0.0, random_init_range,
                                                (self.n_hidden, self.n_input))
        self.weights_ho = np.random.normal(0.0, random_init_range,
                                                (self.n_output, self.n_hidden))

        # Set the learning rate.
        self.lr = learning_rate

    # Use to train the neural network.
    def train(self, inputs_list, targets_list):
        # First convert inputs and targets lists to 2d arrays.
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T
        # Step 1: FEED-FORWARD to get outputs

        # Calculate signals into hidden layer.
        hidden_inputs = np.dot(self.weights_ih, inputs)
        # Calculate the signals emerging from hidden layer.
        hidden_outputs = sigmoid(hidden_inputs)

        # Calculate signals into final output layer.
        final_inputs = np.dot(self.weights_ho, hidden_outputs)
        # Calculate signals emerging from final output layer.
        final_outputs = sigmoid(final_inputs)

        # Step 2: Calculate error

        # Output layer error is the (actual - targets).
        output_errors = final_outputs - targets
        # Hidden layer error is the output_errors, split by weights,
        # recombined at hidden nodes
        hidden_errors = np.dot(self.weights_ho.T, output_errors)

        # Step 3: BACKPROP

        # Derivative of output_errors with respect to final_outputs
        deriv_oe_fo = 2 * output_errors
        # Derivative of final outputs with respect to z2 (weights_ho * hidden_outputs)
        deriv_fo_z2 = sigmoid_derivative(final_outputs)
        # Derivative of z2 with respect to weights_ho
        deriv_z2_who = hidden_outputs.T

        # Update the weights for the links between the hidden and output layers.
        self.weights_ho -= self.lr * np.dot(deriv_oe_fo * deriv_fo_z2, deriv_z2_who)

        # Derivative of hidden_errors with respect to hidden_outputs
        deriv_he_ho = 2 * hidden_errors
        # Derivative of hidden_outputs with respect to z1 (weights_ih * inputs)
        deriv_ho_z1 = sigmoid_derivative(hidden_outputs)
        # Derivative of z1 with respect to weights_ih
        deriv_z1_wih = inputs.T

        # Update the weights for the links between the input and hidden layers.
        self.weights_ih -= self.lr * np.dot(deriv_he_ho * deriv_ho_z1, deriv_z1_wih)


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
