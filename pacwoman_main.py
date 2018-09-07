import gym
import torch.nn as NN
import torch.optim as Optim
from ai import AI
from conv_neural_network import ConvNeuralNetwork
from softmax_body import SoftmaxBody
from experience_replay import NStepProgress, ReplayMemory
from moving_avg import MovingAvg
from eligibility_trace import eligibility_trace
from image_preprocessing import PreprocessImage, create_wrapped_env
from persistance import save_brain, has_save_file, load_brain
from train_ai import train_ai

wrapped_env = create_wrapped_env('MsPacman-v0')
env = PreprocessImage(env = wrapped_env, height = 63, width = 48, grayscale = True)
number_actions = env.action_space.n

env.reset()

conv_network = ConvNeuralNetwork(number_actions)
softmax_body = SoftmaxBody(tempurature = .99)
ai = AI(brain = conv_network, body = softmax_body)

nstep_progress = NStepProgress(env = env, ai = ai, n_step = 10)
replay_memory = ReplayMemory(n_steps = nstep_progress, capacity = 1000000)
moving_avg = MovingAvg(size = 200)

loss = NN.MSELoss()
optimizer = Optim.Adam(conv_network.parameters(), lr = 0.00025)

if (has_save_file()):
    start_epoch = load_brain(conv_network, optimizer, nstep_progress, moving_avg)
else:
    start_epoch = 1

run_count = 10000

train_ai(
    nstep_progress,
    moving_avg, eligibility_trace,
    save_brain, optimizer,
    loss, start_epoch, run_count,
    replay_memory, conv_network
)

env.close()
