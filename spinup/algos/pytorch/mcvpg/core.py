import numpy as np
import scipy.signal
from gym.spaces import Box, Discrete

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical

import math


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input: 
        vector x, 
        [x0, 
         x1, 
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Actor(nn.Module):

    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and 
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class MLPCategoricalActor(Actor):
    
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)


class MLPGaussianActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)    # Last axis sum needed for Torch Normal distribution


class MLPCritic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1) # Critical to ensure v has right shape.



class MLPActorCritic(nn.Module):


    def __init__(self, obs_dim, action_space, 
                 hidden_sizes=(64,64), activation=nn.Tanh):
        super().__init__()

        # policy builder depends on action space
        if isinstance(action_space, Box):
            self.pi = MLPGaussianActor(obs_dim, action_space.shape[0], hidden_sizes, activation)
        elif isinstance(action_space, Discrete):
            self.pi = MLPCategoricalActor(obs_dim, action_space.n, hidden_sizes, activation)

        # build value function
        self.v  = MLPCritic(obs_dim, hidden_sizes, activation)

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]

# MCTS
class Node:
    """
    Tree Node for MCTS
    """
    def __init__(self, parent=None, children=None):
        self.parent = parent
        self.children = children or []
        self.score = 0
        self.visited_count = 0
        self.leaf_node = False
        self.state = {}

    def copy_env(self, env):
        self.state['agent_pos'] = env.agent_pos
        self.state['agent_dir'] = env.agent_dir
        self.state['step_count'] = env.step_count
    
    def restore_env(self, env):
        env.agent_pos = self.state['agent_pos']
        env.agent_dir = self.state['agent_dir']
        env.step_count = self.state['step_count']

    def update_score(self, s):
        self.score = (self.score * self.visited_count + s) / (self.visited_count + 1) 
        self.visited_count += 1

    def get_score(self):
        return self.score
    
    def is_leaf_node(self):
        return self.leaf_node

    def get_UCB_score(self, C=0.5):
        return self.score + C * math.sqrt (math.log(self.parent.visited_count) / self.visited_count)
    
    
    @staticmethod
    def MCTS(env, search_count=1000, max_depth=5):
        root = Node()
        root.copy_env(env)
        actions = [i for i in range(env.action_space.n)]
        cur_search_count = 0
        while(cur_search_count < search_count):
            cur_search_count += 1

            root.restore_env(env)
            cur_node = root
            cur_depth = 1

            # Selection
            while not cur_node.is_leaf_node() and (cur_depth < max_depth):
                if len(cur_node.children) < len(actions):
                    a = actions[len(cur_node.children)]
                    __node = Node(parent=cur_node)
                    cur_node.children.append(__node)
                    
                    _, r, done, info = env.step(a)
                    cur_node = __node

                    if done:
                        __node.leaf_node = True
                        __node.score = r
                    elif info.get('useless'):
                        __node.leaf_node = True
                        __node.score = -1.0
                    break
                else:
                    # UCB
                    cur_depth += 1
                    a = np.argmax([n.get_UCB_score() for n in cur_node.children])
                    __node = cur_node.children[a]
                    _, r, done, info = env.step(a)
                    cur_node = __node
                
            # Back-propagation
            if cur_node.is_leaf_node():
                score = cur_node.get_score()
            else:
                # score = Node.random_evaluation(env, max_steps=5, eva_epoches=50)
                # score = Node.quick_evaluation(env)
                score = Node.random_quick_evaluation(env, max_steps=5, eva_epoches=20)
            while (cur_node != None):
                cur_node.update_score(score)
                cur_node = cur_node.parent

            root.restore_env(env)
            scores = [n.get_score() for n in root.children]
            a = np.argmax(scores)
        # In sequential MCTS, don't delete the root, shift it to the child node
        del root
        return a

    @staticmethod
    def random_evaluation(env, max_steps=5, eva_epoches=50):
        avg = []
        __node = Node()
        __node.copy_env(env)
        for _ in range(eva_epoches):
            __node.restore_env(env)
            actions = [i for i in range(env.action_space.n)]
            done = False
            r = 0.0
            step = 0
            while not done and (step < max_steps):
                step += 1
                a = np.random.randint(0, env.action_space.n)
                _, r, done, _ = env.step(a)
                avg.append(r)
        __node.restore_env(env)
        return np.mean(avg)
    @staticmethod
    def random_quick_evaluation(env, max_steps=5, eva_epoches=50):
        avg = []
        __node = Node()
        __node.copy_env(env)
        for _ in range(eva_epoches):
            __node.restore_env(env)
            actions = [i for i in range(env.action_space.n)]
            done = False
            r = 0.0
            step = 0
            while not done and (step < max_steps):
                step += 1
                a = np.random.randint(0, env.action_space.n)
                _, _, done, _ = env.step(a)
            avg.append(Node.quick_evaluation(env))
        __node.restore_env(env)
        return np.mean(avg)

    @staticmethod
    def quick_evaluation(env):
        max_dis = env.grid.width*env.grid.width + env.grid.height*env.grid.height
        goal_pos = env.grid.find('goal')[0]
        g_x, g_y = goal_pos
        a_x, a_y = env.agent_pos
        cur_dis = (g_x-a_x)*(g_x-a_x)+(g_y-a_y)*(g_y-a_y)
        return (max_dis-cur_dis)/max_dis