import sys
import gym
import torch.nn as NN
import torch.optim as Optim
from torch.autograd import Variable
from ai import AI
from conv_neural_network import ConvNeuralNetwork
from softmax_body import SoftmaxBody
from experience_replay import NStepProgress, ReplayMemory
from moving_avg import MovingAvg
from eligibility_trace import eligibility_trace
from image_preprocessing import PreprocessImage, create_wrapped_env

wrapped_env = create_wrapped_env('MsPacman-v0')
env = PreprocessImage(env = wrapped_env, height = 63, width = 48, grayscale = True)
number_actions = env.action_space.n

env.reset()

conv_network = ConvNeuralNetwork(number_actions)
softmax_body = SoftmaxBody(tempurature = 8.0)
ai = AI(brain = conv_network, body = softmax_body)

nstep_progress = NStepProgress(env = env, ai = ai, n_step = 15)
replay_memory = ReplayMemory(n_steps = nstep_progress, capacity = 10000)
moving_avg = MovingAvg(size = 200)

loss = NN.MSELoss()
optimizer = Optim.Adam(conv_network.parameters(), lr = 0.001)

run_count = 10000

for run_num in range(1, run_count + 1):
    replay_memory.run_steps(200)

    for batch in replay_memory.sample_batch(128):
        inputs, targets = eligibility_trace(conv_network, batch)
        inputs, targets = Variable(inputs), Variable(targets)
        predictions = conv_network(inputs)
        loss_error = loss(predictions, targets)

        optimizer.zero_grad()
        loss_error.backward()
        optimizer.step()

    rewards_steps = nstep_progress.rewards_steps()

    moving_avg.add(rewards_steps)

    avg_reward = moving_avg.average()

    print("Run: %s, Average Reward: %s" % (str(run_num), str(avg_reward)))

env.close()
