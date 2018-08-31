import torch.nn as NN
import torch.nn.functional as Functional

class SoftmaxBody(NN.Module):
    def __init__(self, tempurature):
        super(SoftmaxBody, self).__init__()
        self.tempurature = tempurature

    def forward(self, outputs):
        probs = Functional.softmax(outputs * self.tempurature, dim = 1)
        actions = probs.multinomial(1)

        return actions
