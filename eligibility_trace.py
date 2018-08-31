import torch
import numpy as NP
from torch.autograd import Variable

def __from_array__(float_inputs):
    return torch.from_numpy(NP.array(float_inputs, dtype = NP.float32))

def __process_batch__(conv_network, series):
    input = NP.array([series[0].state, series[-1].state], dtype = NP.float32)
    input = Variable(torch.from_numpy(input))
    output = conv_network(input)
    total_reward = 0.0 if series[-1].done else output[1].data.max()

    return (output, total_reward)

def eligibility_trace(conv_network, batch, gamma = 0.99):
    inputs = []
    targets = []

    for series in batch:
        output, batch_reward = __process_batch__(conv_network, series)

        for step in reversed(series[:-1]):
            batch_reward = step.reward + gamma * batch_reward

        state = series[0].state
        target = output[0].data
        target[series[0].action] = batch_reward

        inputs.append(state)
        targets.append(target)

    return __from_array__(inputs), torch.stack(targets)
