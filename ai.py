import numpy as np
import torch
from torch.autograd import Variable

class AI:
    def __init__(self, brain, body):
        self.brain = brain
        self.body = body

    def __call__(self, inputs):
        input = Variable(torch.from_numpy(np.array(inputs, dtype = np.float32)))
        output_possibilities = self.brain(input)
        chosen_possibilies = self.body(output_possibilities)

        return chosen_possibilies.data.numpy()
