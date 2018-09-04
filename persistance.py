import torch
from os.path import isfile

save_file = 'save.pth.tar'

def save_brain(epoch, conv_network, optimizer, nstep_progress, moving_avg):
    torch.save({
        'epoch': epoch + 1,
        'nstep_rewards': nstep_progress.rewards,
        'moving_avg': moving_avg.rewards_history,
        'state_dict': conv_network.state_dict(),
        'optimizer': optimizer.state_dict()
    }, save_file)

def has_save_file():
    return isfile(save_file)

def load_brain(conv_network, optimizer, nstep_progress, moving_avg):
    checkpoint = torch.load(save_file)

    conv_network.load_state_dict(checkpoint['state_dict'])
    conv_network.load_state_dict(checkpoint['optimizer'])
    moving_avg.initilize_rewards_history(checkpoint['moving_avg'])
    nstep_progress.initilize_rewards(checkpoint['nstep_rewards'])

    return checkpoint['epoch']
