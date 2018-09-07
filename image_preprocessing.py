# Image Preprocessing

# Importing the libraries
import numpy as np
from scipy.misc import imresize
from gym import make
from gym.core import ObservationWrapper
from gym.wrappers import Monitor
from gym.spaces.box import Box
from skip_wrapper import SkipWrapper

# Preprocessing the Images

class PreprocessImage(ObservationWrapper):
    def __init__(self, env, height = 64, width = 64, grayscale = True, crop = lambda img: img):
        super(PreprocessImage, self).__init__(env)
        self.img_size = (height, width)
        self.grayscale = grayscale
        self.crop = crop
        n_colors = 1 if self.grayscale else 3
        self.observation_space = Box(0.0, 1.0, [n_colors, height, width], dtype = np.float32)

    def observation(self, img):
        img = self.crop(img)
        img = imresize(img, self.img_size)
        if self.grayscale:
            img = img.mean(-1, keepdims = True)
        img = np.transpose(img, (2, 0, 1))
        img = img.astype('float32') / 255.
        return img

def create_wrapped_env(
    game_name,
    skip_count = 4,
    video_callable_episode_id = lambda episode_id: episode_id % 1 == 0
):
    skip_wrapper = SkipWrapper(skip_count)

    return skip_wrapper(Monitor(
        make(game_name),
        "./videosv2/",
        mode = 'training',
        resume = True,
        video_callable = video_callable_episode_id
    ))
