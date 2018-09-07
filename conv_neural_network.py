import torch
import torch.nn as NN
import torch.nn.functional as Functional
from torch.autograd import Variable
from math import ceil

def create_conv_network(in_channels, out_channels, kernel_size):
    return NN.Conv2d(
        in_channels = in_channels,
        out_channels = out_channels,
        kernel_size = kernel_size
    )

def create_full_connection(in_features, out_features):
    return NN.Linear(in_features = in_features, out_features = out_features)

class ConvNeuralNetwork(NN.Module):
    def __init__(self, num_actions):
        super(ConvNeuralNetwork, self).__init__()

        self.conv_layer_1 = create_conv_network(1, 32, 7)
        self.conv_layer_2 = create_conv_network(32, 32, 5)
        self.conv_layer_3 = create_conv_network(32, 64, 2)

        num_neurons = self.count_neurons((1, 60, 64))
        self.neural_hidden_layer_1 = create_full_connection(num_neurons, num_neurons)

        output_input_neurons = ceil((num_neurons + num_actions) / 2)

        self.neural_hidden_layer_2 = create_full_connection(num_neurons, output_input_neurons)
        self.neural_output_layer = create_full_connection(output_input_neurons, num_actions)

    def count_neurons(self, image_dim):
        x = Variable(torch.rand(1, *image_dim))

        return self.activate_conv_network(x).data.view(1, -1).size(1)

    def activate_conv_network(self, x):
        x = Functional.relu(Functional.max_pool2d(self.conv_layer_1(x), 3, 2))
        x = Functional.relu(Functional.max_pool2d(self.conv_layer_2(x), 3, 2))

        return Functional.relu(Functional.max_pool2d(self.conv_layer_3(x), 3, 2))

    def forward(self, input_data):
        x = self.activate_conv_network(input_data)
        x = x.view(x.size(0), -1)
        x = Functional.relu(self.neural_hidden_layer_1(x))
        x = Functional.relu(self.neural_hidden_layer_2(x))
        output_neurons = self.neural_output_layer(x)

        return output_neurons
