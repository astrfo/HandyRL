# Copyright (c) 2020 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

# OpenAI Gym

import os
import random
from collections import deque

import gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from ..environment import BaseEnvironment


def get_screen(env):
    resize = T.Compose([T.ToPILImage(),
                    T.Resize(size=(84, 84)),
                    T.Grayscale(num_output_channels=1)])
    screen = resize(env.render())
    screen = np.expand_dims(np.asarray(screen), axis=2).transpose((2, 0, 1))
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    return screen


class SimpleConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        C, H, W = 1, 84, 84
        self.kernel_sizes = [8, 4, 3]
        self.strides = [4, 2, 1]
        nn_size = 512
        neighbor_frames = 4

        self.conv1 = nn.Conv2d(in_channels=C*neighbor_frames, out_channels=16, kernel_size=self.kernel_sizes[0], stride=self.strides[0])
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=self.kernel_sizes[1], stride=self.strides[1])
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=self.kernel_sizes[2], stride=self.strides[2])
        self.bn3 = nn.BatchNorm2d(32)

        convw = self.conv2d_size_out(W, n_layer=3)
        convh = self.conv2d_size_out(H, n_layer=3)
        linear_input_size = convw * convh * 32
        self.fc1 = nn.Linear(linear_input_size, nn_size)
        self.head_p = nn.Linear(nn_size, 4)
        self.head_v = nn.Linear(nn_size, 1)

    def forward(self, x, hidden=None):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        h_p = self.head_p(x)
        h_v = self.head_v(x)
        return {'policy': h_p, 'value': torch.tanh(h_v)}

    def conv2d_size_out(self, size, n_layer):
        cnt = 0
        size_out = size
        while cnt < n_layer:
            size_out = (size_out - self.kernel_sizes[cnt]) // self.strides[cnt] + 1
            cnt += 1
        return size_out


class Environment(BaseEnvironment):
    def __init__(self, args={}):
        super().__init__()
        self.env = gym.make('Breakout-v4', render_mode='rgb_array').unwrapped
        self.neighbor_frames = 4
        self.frames = None
        self.latest_obs = None
        self.total_reward = 0
        self.done, self.truncated = False, False

    def reset(self, args={}):
        self.env.reset()
        frame = get_screen(self.env)
        self.frames = deque([frame]*self.neighbor_frames, maxlen=self.neighbor_frames)
        self.latest_obs = np.stack(self.frames, axis=1)[0,:]
        self.total_reward = 0
        self.done, self.truncated = False, False

    def play(self, action, player):
        observation, reward, terminal, truncated, info = self.env.step(action)
        frame = get_screen(self.env)
        self.frames = deque([frame]*self.neighbor_frames, maxlen=self.neighbor_frames)
        self.latest_obs = np.stack(self.frames, axis=1)[0,:]
        self.latest_reward = reward
        self.done = terminal
        self.truncated = truncated
        self.latest_info = info
        self.total_reward += reward

    def terminal(self):
        return self.done or self.truncated

    def outcome(self):
        outcomes = [self.total_reward]
        return {p: outcomes[idx] for idx, p in enumerate(self.players())}

    def players(self):
        return [0]

    def net(self):
        return SimpleConvNet()

    def observation(self, player=None):
        return self.latest_obs

    def action_length(self):
        return self.env.action_space.n

    def legal_actions(self, player=None):
        return list(range(self.action_length()))


if __name__ == '__main__':
    e = Environment()
    for _ in range(100):
        print(_)
        e.reset()
        while not e.terminal():
            e.env.render()
            actions = e.legal_actions()
            e.play(random.choice(actions))
    e.env.close()