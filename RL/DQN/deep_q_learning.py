# This is recommended skleton code. You can change this file as you want. 

from pickletools import optimize
import numpy as np
np.random.seed(1)

import random
from tqdm import tqdm
from collections import deque, defaultdict
import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Recommended hyper-parameters. You can change if you want #



class DeepQLearning:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            discount_factor=0.9,
            e_greedy=0.05,
            replace_target_iter=300,
            memory_size=500,
            batch_size=32
    ):

    ######################################## TO DO ##############################################
    # Initialzie variables here
        self.Q_table = defaultdict(lambda: np.zeros(4)) # Q_table = {"state":(Q(s,UP),Q(s,DOWN),Q(s,LEFT),Q(s,RIGHT),)}
        self.n_actions = n_actions
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.e_greedy = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size

        self.memory = deque(maxlen=self.memory_size)

        self.model = nn.Sequential(
            nn.Linear(self.n_features, 64),
            nn.ReLU(),
            nn.Linear(64, self.n_actions)
        )
        self.optimizer = optim.Adam(self.model.parameters(), self.learning_rate)

        self.e_start = 0.9
        self.e_dec = 200
        self.steps_done = 0


    def store_transition(self, s, a, r, next_s):
        self.memory.append((s, a, torch.FloatTensor([r]), torch.FloatTensor([next_s]))) 

        ######################################## TO DO ##############################################
    def choose_action(self, state):
        sample = random.random()
        #eps_threshold = self.e_greedy
        eps_threshold = self.e_greedy + (self.e_start - self.e_greedy) * math.exp(-1. * self.steps_done / self.e_dec)
        self.steps_done += 1

        if sample > eps_threshold:
            return self.model(state).data.max(1)[1].view(1,1)  # o-dimension
        else:
           return torch.LongTensor([[random.randrange(self.n_actions)]])
        ######################################## TO DO ##############################################
    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        mini_batch = random.sample(self.memory, self.batch_size)
        state, action, reward, next_state = zip(*mini_batch) 
        states = torch.cat(state)
        actions = torch.cat(action)
        rewards = torch.cat(reward)
        next_states = torch.cat(next_state)
        current_q = self.model(states).gather(1, actions)
        max_next_q = self.model(next_states).detach().max(1)[0]
        expected_q = rewards + (self.discount_factor * max_next_q)
        loss = F.mse_loss(current_q.squeeze(), expected_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        ######################################## TO DO ##############################################
