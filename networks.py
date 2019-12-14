import os
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.distributions import Categorical, Normal, MultivariateNormal
import pdb

from utils import *
from collections import namedtuple
import random
import gym

device='cpu'

# https://github.com/openai/gym/blob/master/gym/core.py
class NormalizedEnv(gym.ActionWrapper):
    """ Action wrapper than normalizes input actions"""
    def _action(self, action):
        act_k = (self.action_space.high - self.action_space.low)/ 2.
        act_b = (self.action_space.high + self.action_space.low)/ 2.
        return act_k * action + act_b

    def _reverse_action(self, action):
        act_k_inv = 2./(self.action_space.high - self.action_space.low)
        act_b = (self.action_space.high + self.action_space.low)/ 2.
        return act_k_inv * (action - act_b)


# Transition = namedtuple('Transition',('full_state','state', 'next_state', 'action', 'reward'))
Transition = namedtuple('Transition',('state', 'next_state', 'action', 'reward'))

# Using PyTorch's Tutorial
class ReplayMemory(object):
    '''A class for keeping track of collected transitions from interacting with the environment'''

    def __init__(self, size):
        self.size = size
        self.memory = []
        self.temp_memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.size:
            self.memory.append(None)

        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.size

    def temp_push(self, *args):
        '''saves a transition temporarily'''
        self.temp_memory.append(Transition(*args))

    def clear_temp(self):
        '''clears temporarily saved transitions'''
        self.temp_memory = []

    def clear(self):
        '''clears the memory'''
        self.memory = []
        self.position = 0

    def sample(self, batch_size, temp_only=False):
        '''
        sample batch_size transitions from the memory, if temp_only = True, only
        sample from temp_memory. Otherwise sample from both temp_memory and memory

        Args:
            batch_size: number of transitions to sample from memory or temp_memory
            temp_only: (bool) usage above

        Returns:
            randomly sampled batch_size number of transitions from stored transitions
        '''
        if temp_only and len(self.temp_memory) >= batch_size:
            return random.sample(self.temp_memory, batch_size)
        else:
            return random.sample(self.memory + self.temp_memory, batch_size)

    def __len__(self):
        return len(self.memory)


class Policy(nn.Module):
    '''A class for stochastic neural-network policies. The case for discrete action-spaces needs more testing'''
    def __init__(self, in_dim, out_dim, continuous=False, std=-0.8, max_torque=1., small=False):#-0.8 GOOD
        '''
        Args:
            in_dim: dimension of policy observations/states
            out_dim: dimension of actions
            continuous: True if action_space is continuous, False if discrete
            std: may be used (along with uncommenting some lines below) to keep the standard deviation of policy fixed,
                it is learnable by default
            max_torque: maximum value of action allowed
            small: True if should use a smaller version of the policy (hidden size=8), otherwise hidden_size=64
        '''
        super(Policy, self).__init__()
        self.n_actions = out_dim
        self.continuous = continuous
        self.max_torque = max_torque
        self.small = small
        if not small:
            self.lin1 = nn.Linear(in_dim, 64)
            self.relu = nn.ReLU()
            self.theta = nn.Linear(64, 64)
            self.action_head = nn.Linear(64,out_dim)
            torch.nn.init.xavier_uniform_(self.theta.weight)
        else:
            self.lin1 = nn.Linear(in_dim, 8)
            self.relu = nn.ReLU()
            self.action_head = nn.Linear(8,out_dim)
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        if continuous:
            if not small:
                self.log_std = nn.Linear(64, out_dim)
                # self.log_std = nn.Parameter(torch.ones(out_dim) * std, requires_grad=True)
            else:
                self.log_std = nn.Linear(8, out_dim)

    def forward(self, x):
        x = self.relu(self.lin1(x))
        if not self.small:
            x = self.relu(self.theta(x))
        return x

    def sample_action(self, x):
        '''
        Sample an action according to probability distribution calculated using policy given x
        Args:
            x: batch of states or observations on which the policy conditions on

        Returns:
            batch of actions sampled from a distribution calculated using the policy params conditioned on x
        '''
        action_probs = self.get_action_probs(x)
        if not self.continuous:
            c = Categorical(action_probs[0])
            a = c.sample()
        else:
            c = Normal(*action_probs)
            #a = torch.clamp(c.rsample(), min=-self.max_torque, max=self.max_torque)
            a = c.rsample()#self.action_multiplier * c.rsample()
        return a

    def get_action_probs(self, x):
        '''Calculate probability distribution of actions given batch input x
        Args:
            x: batch of states

        Returns:
            tuple of (mean, standard dev) of the probability distribution of policy if continuous
            if discrete, return (softmax, 0)
        '''
        if self.continuous:
            mu = nn.Tanh()(self.action_head(self.forward(x)))*self.max_torque
            sigma_sq = F.softplus(self.log_std(self.forward(x))) + 0.1
            # sigma_sq = F.softplus(self.log_std.exp()).expand_as(mu)
            self.sigma_sq = sigma_sq.mean(dim=0).detach()
            return (mu,sigma_sq)
        else:
            return (nn.Softmax(dim=-1)(self.action_head(self.forward(x))),0)


class DeterministicPolicy(nn.Module):
    '''A class for a neural-network policy that is deterministic'''
    def __init__(self, in_dim, out_dim, max_action):
        '''
        Args:
            in_dim: dimension of observations/states input to policy
            out_dim: dimension of actions
            max_action: maximum value of allowable action to take
        '''
        super(DeterministicPolicy, self).__init__()
        self.n_actions = out_dim
        self.max_action = max_action
        self.p_type = 'nn'
        self.lin1 = nn.Linear(in_dim, 30)
        # self.lin1 = nn.Linear(in_dim, out_dim)
        self.relu = nn.ReLU()
        self.theta = nn.Linear(30, 30)
        self.action_head = nn.Linear(30,out_dim)

        torch.nn.init.xavier_uniform_(self.lin1.weight)
        torch.nn.init.xavier_uniform_(self.theta.weight)
        torch.nn.init.xavier_uniform_(self.action_head.weight)

    def forward(self, x):
        x = self.relu(self.lin1(x))
        x = self.relu(self.theta(x))
        x = nn.Tanh()(self.action_head(x))
        # x = nn.Tanh()(self.lin1(x))
        return x

    def sample_action(self, x):
        ''' Return an action given observation batch x. Since policy is deterministic, only need to calculate forward'''
        action = self.forward(x) * self.max_action
        return action


class Value(nn.Module):
    '''A class for a neural-network critic'''
    def __init__(self, states_dim, actions_dim, pretrain_val_lr=1e-4, pretrain_value_lr_schedule=None):
        super(Value, self).__init__()
        self.lin1 = nn.Linear(states_dim+actions_dim, 64)
        self.lin2 = nn.Linear(64, 64)
        self.relu = nn.ReLU()
        self.theta = nn.Linear(64, 1)

        torch.nn.init.xavier_uniform_(self.lin1.weight)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        torch.nn.init.xavier_uniform_(self.theta.weight)

        self.q_optimizer = optim.Adam(self.parameters(), lr=pretrain_val_lr)
        #this learning rate schedule is not used but may be implemented if necessary in the future
        self.pretrain_value_lr_schedule = None
        self.states_dim = states_dim
        self.actions_dim = actions_dim

    def reset_weights(self):
        '''Reset the weights of the network'''
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        torch.nn.init.xavier_uniform_(self.theta.weight)

    def forward(self, x, u):
        xu = torch.cat([x, u], 1)
        x = self.relu(self.lin1(xu))
        values = self.theta(self.relu(self.lin2(x)))
        return values

    def pre_train(self, actor, dataset, epochs_value, discount, batch_size, salient_states_dim, file_location, file_id,
                  model_type, env_name, max_actions, verbose=100):
        '''Pretrains the critic using policy actor on data collected from the true environment.

        Args:
            actor: policy whose value we'd like to evaluate
            dataset (ReplayMemory): the collection of transitions sampled from the environment
            epochs_value (int): number of iterations to train critic
            discount (float): between 0 and 1
            batch_size (int)
            salient_states_dim (int): dimension of relevant state dimensions
            file_location (str): directory for saving model checkpoints and training stats
            file_id (str): string to append to end of file names for saving
            model_type (str): either 'mle', 'paml', or 'random' for no training at all. Not used in this function, but
                            specifies how the calling function is using this (passed as arg for bookkeeping)
            env_name (str): name of environment
            max_actions (int): Number of actions in a trajectory (= number of states - 1)
            verbose (int)

        Returns:
            None
        '''
        MSE = nn.MSELoss()
        TAU=0.001
        target_critic = Value(self.states_dim, self.actions_dim).double()

        for target_param, param in zip(target_critic.parameters(), self.parameters()):
            target_param.data.copy_(param.data)

        train_losses = []
        for i in range(epochs_value):
            batch = dataset.sample(batch_size)
            states_prev = torch.tensor([samp.state for samp in batch]).double().to(device)
            states_next = torch.tensor([samp.next_state for samp in batch]).double().to(device)
            rewards_tensor = torch.tensor([samp.reward for samp in batch]).double().to(device).unsqueeze(1)
            actions_tensor = torch.tensor([samp.action for samp in batch]).double().to(device)
            # actions_next = actor.sample_action(states_next[:,:salient_states_dim])
            actions_next = actor.sample_action(states_next)#[:,:salient_states_dim])
            # target_q = target_critic(states_next[:,:salient_states_dim], actions_next)
            target_q = target_critic(states_next, actions_next)
            y = rewards_tensor + discount * target_q.detach() #detach to avoid backprop target
            # q = critic(states_prev[:,:salient_states_dim], actions_tensor)
            q = self(states_prev, actions_tensor)

            self.q_optimizer.zero_grad()
            loss = MSE(y, q)
            train_losses.append(loss.detach().data)

            if i % verbose == 0:
                print('Epoch: {:4d} | LR: {:.4f} | Value estimator loss: {:.5f}'.format(
                    i, self.q_optimizer.param_groups[0]['lr'], loss.detach().cpu()))
                torch.save(self.state_dict(), os.path.join(
                    file_location,'critic_policy_{}_state{}_salient{}_checkpoint_{}_traj{}_{}.pth'.format(
                        model_type, self.states_dim, salient_states_dim, env_name, max_actions + 1, file_id)))
            loss.backward()
            nn.utils.clip_grad_value_(self.parameters(), 100.0)
            self.q_optimizer.step()
            if self.pretrain_value_lr_schedule is not None:
                self.pretrain_value_lr_schedule.step()
            #soft update the target critic
            for target_param, param in zip(target_critic.parameters(), self.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - TAU) + param.data * TAU)
            # if non_decreasing(train_losses):
            # 	self.q_optimizer = optim.Adam(self.parameters(), lr=self.q_optimizer.param_groups[0]['lr']/2)
        del target_critic


# Ornstein-Ulhenbeck Process
# Taken from #https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py
class OUNoise(object):
    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000):
        self.mu           = mu
        self.theta        = theta
        self.sigma        = max_sigma
        self.max_sigma    = max_sigma
        self.min_sigma    = min_sigma
        self.decay_period = decay_period
        self.action_dim   = action_space.shape[0]
        self.low          = action_space.low
        self.high         = action_space.high
        self.reset()
        
    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu
        
    def evolve_state(self):
        x  = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state
    
    def get_action(self, action, t=0, multiplier=1.0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + multiplier*ou_state, self.low, self.high)

