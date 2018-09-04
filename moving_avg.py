import numpy as NP

class MovingAvg:
    def __init__(self, size):
        self.rewards_history = []
        self.size = size

    def initilize_rewards_history(rewards_history):
        self.rewards_history = rewards_history

    def add(self, rewards):
        if isinstance(rewards, list):
            self.rewards_history += rewards
        else:
            self.rewards_history.append(rewards)

        while len(self.rewards_history) > self.size:
            del self.rewards_history[0]

    def average(self):
        return NP.mean(self.rewards_history)
