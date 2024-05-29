from collections import deque 
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from args import parse_args 
import time

args = parse_args()
np.random.seed(args.seed)
torch.manual_seed(args.seed)

class Deque(deque):
    """
    This class is used to store training losses and Q-values of neural nets. 
    """
    def __init__(self, max_size):
        """
        Parameters:
            max_size (int): maximum size of the double-ended queue (deque)
        """
        super().__init__()
        self.max_size = max_size 

    def append(self, item):
        """ 
        Parameters:
            item (float): item to be added
        """
        if len(self) >= self.max_size:
            self.popleft()
        super().append(item)
    
    def is_full(self):
        """
        Returns:
            (bool): True if the deque is full
        """
        return len(self) == self.max_size
    
class PPOMemory:
    """
    This class is used to store and retrieve experiences for training a policy with
    the PPO algorithm.
    """
    def __init__(self, mini_batch_size):
        """
        Parameters:
            n_observations (int): number of observations
            n_actions (int): number of actions
            batch_size (int): maximum size of the memory
            mini_batch_size (int): number of elements in the mini-batches
        """

        self.mini_batch_size = mini_batch_size 

        self.observations = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

    def store_memory(self, observation, action, prob, val, reward, done):
        self.observations.append(observation)
        self.actions.append(action)
        self.probs.append(prob)
        self.vals.append(val)
        self.rewards.append(reward)
        self.dones.append(done)

    def generate_batches(self):
        n_observations = len(self.observations)
        batch_start = np.arange(0, n_observations, self.mini_batch_size)
        indices = np.arange(n_observations, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.mini_batch_size] for i in batch_start]
        return np.array(self.observations),\
                np.array(self.actions),\
                np.array(self.probs),\
                np.array(self.vals),\
                np.array(self.rewards),\
                np.array(self.dones),\
                batches

    def clear_memory(self):
        self.observations = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []

class Actor(nn.Module):
    """
    This class defines the architecture and forward pass of the actor network in PPO.
    """
    def __init__(self, n_observations, n_actions, lr=args.learning_rate, 
                 fc1_dim=args.fc1_dims, fc2_dim=args.fc2_dims):
        """
        Initializing the actor network

        Parameters:
            n_observations (int): number of observations
            n_actions (int): number of actions 
            lr (float): learning rate for the optimizer 
            fc1_dim (int): number of neurons in the first fully connected layer 
            fc2_dim (int): number of neurons in the second fully connected layer 
        """
        super(Actor, self).__init__()

        self.actor = nn.Sequential(
            self.actor_layer_init(nn.Linear(n_observations, fc1_dim)),
            nn.Tanh(),
            self.actor_layer_init(nn.Linear(fc1_dim, fc2_dim)),
            nn.Tanh(),
            self.actor_layer_init(nn.Linear(fc2_dim, n_actions), std=0.01)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def actor_layer_init(self, layer, std=np.sqrt(2), bias=0):
        """
        Weight and bias initialization for a given actor network layer 

        Parameters:
            layer (torch.nn.Module): layer to be initialized 
            std (float): standard deviation for weight initialization 
            bias (float): constant value for bias initialization 

        Returns:
            (torch.nn.Module): initialized layer
        """
        torch.nn.init.orthogonal_(layer.weight, std) # orthogonal 
        torch.nn.init.constant_(layer.bias, bias) # bias
        return layer 

    def forward(self, observation):
        """
        Forward pass of the actor network

        Parameters: 
            observation (torch.Tensor): tensor of observed values 
        
        Returns:
            logits (torch.Tensor): output of the actor network
        """
        logits = self.actor(observation)
        return logits 

class Critic(nn.Module):
    """
    This class defines the architecture and forward pass of the critic network in PPO.
    Outputs are approximating the true value function. 
    """
    def __init__(self, n_observations, lr=args.learning_rate,
                 fc1_dim=args.fc1_dims, fc2_dim=args.fc2_dims):
        """
        Initializing the critic network

        Parameters:
            n_observations (int): number of observations
            lr (float): learning rate for the optimizer 
            fc1_dim (int): number of neurons in the first fully connected layer 
            fc2_dim (int): number of neurons in the second fully connected layer 
        """
        super(Critic, self).__init__()

        self.critic = nn.Sequential(
            self.critic_layer_init(nn.Linear(n_observations, fc1_dim)),
            nn.Tanh(),
            self.critic_layer_init(nn.Linear(fc1_dim, fc2_dim)),
            nn.Tanh(),
            self.critic_layer_init(nn.Linear(fc2_dim, 1), std=1.0)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def critic_layer_init(self, layer, std=np.sqrt(2), bias=0):
        """
        Weight and bias initialization for a given critic network layer 

        Parameters:
            layer (torch.nn.Module): layer to be initialized 
            std (float): standard deviation for weight initialization 
            bias (float): constant value for bias initialization 

        Returns:
            (torch.nn.Module): initialized layer
        """
        torch.nn.init.orthogonal_(layer.weight, std) # orthogonal 
        torch.nn.init.constant_(layer.bias, bias) # bias
        return layer 

    def forward(self, observation):
        """
        Forward pass of the critic network

        Parameters: 
            observation (torch.Tensor): tensor of observed values 
        
        Returns:
            value (torch.Tensor): output of the critic network
        """
        value = self.critic(observation)
        return value
    

class PPONet(nn.Module):
    """
    Proximal Policy Optimization network model
    """
    def __init__(self, n_observations, n_actions, n_epochs=args.update_epochs, 
                 minibatch_size=args.minibatch_size, gamma=args.gamma, 
                 gae_lambda=args.gae_lambda, epsilon_clip=args.eps_clip):
        """
        Initializing the PPO network model

        Parameters:
            n_observations (int): number of observations
            n_actions (int): number of actions
            n_epochs (int): number of training epochs
            batch_size (int): maximum size of the memory
            minibatch_size (int): number of elements in the mini-batches
            gamma (float): discount factor future rewards 
            gae_lambda (float): for the general advantage estimate 
            epsilon_clip (float): for PPO-clipping
        """
        super(PPONet, self).__init__()
        self.n_epochs = n_epochs 
        self.gamma = gamma 
        self.gae_lambda = gae_lambda
        self.eps_clip = epsilon_clip

        self.critic = Critic(n_observations)
        self.actor = Actor(n_observations, n_actions)
        self.memory = PPOMemory(minibatch_size)

    def store(self, observation, action, prob, val, reward, done):
        """
        Stores an experience in the memory

        Parameters:
            observation ():
            action ():
            prob ():
            val ():
            reward ():
            done ():
        """
        self.memory.store_memory(observation, action, prob, val, reward, done)

    def choose_action(self, obs):
        """
        Selects an action given the current policy 

        Parameters:
            obs (numpy.ndarray): array of current observation  

        Returns:
            action (): action chosen 
            logprob (float): log probability of chosen action 
            value (float): estimated value of the state (from critic)
        """
        observation = torch.tensor(obs, device=self.actor.device, dtype=torch.float).unsqueeze(0)
        
        logits = self.actor(observation)
        dist = Categorical(logits = logits) # categorical distribution 
        action = dist.sample()
        logprob = dist.log_prob(action)
        value = self.critic(observation)

        return action, logprob, value
    
    def normalized_advantages(self, advantages):
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages
    
    def compute_critic_loss_clipped(self, returns, critic_value, value):
        c_loss_unclipped = (returns-critic_value)**2
        c_clipped = value + torch.clamp(critic_value - value, -args.eps_clip, args.eps_clip)
        c_loss_clipped = (c_clipped - returns)**2
        loss_max = torch.max(c_loss_unclipped, c_loss_clipped)
        return 0.5*loss_max.mean()

    def update(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr,\
            reward_arr, dones_arr, batches = \
                    self.memory.generate_batches()

            values = vals_arr

            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr)-1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr)-1):
                    a_t += discount*(reward_arr[k] + self.gamma*values[k+1]*\
                            (1-int(dones_arr[k])) - values[k])
                    discount *= self.gamma*self.gae_lambda
                advantage[t] = a_t
            advantage = torch.tensor(advantage).to(self.actor.device)

            values = torch.tensor(values).to(self.actor.device)
            for batch in batches:
                states = torch.tensor(state_arr[batch], dtype=torch.float).to(self.actor.device)
                old_probs = torch.tensor(old_prob_arr[batch]).to(self.actor.device)
                actions = torch.tensor(action_arr[batch]).to(self.actor.device)
                

                logits = self.actor(states)
                dist = Categorical(logits = logits)
                critic_value = self.critic(states).squeeze()

                # actor_loss_clip
                new_probs = dist.log_prob(actions)
                prob_ratio = (new_probs - old_probs).exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = torch.clamp(prob_ratio, 1-self.eps_clip,
                        1+self.eps_clip)*advantage[batch]
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()
                returns = advantage[batch] + values[batch]

                # # critic_loss_clip 
                # critic_loss_unclipped = (returns[batch]-critic_value)**2
                # critic_clipped = values[batch] + torch.clamp(critic_value - values[batch], -args.eps_clip, args.eps_clip)
                # critic_loss_clipped = (critic_clipped - returns[batch]) ** 2

                # critic_loss_max = torch.max(critic_loss_unclipped, critic_loss_clipped)
                # critic_loss = 0.5 * critic_loss_max.mean()
                critic_loss = (returns-critic_value)**2
                critic_loss = 0.5*critic_loss.mean()
        

                total_loss = actor_loss + 0.5*critic_loss

                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.memory.clear_memory()    
    
        return total_loss.item(), actor_loss.item(), critic_loss.item(), torch.mean(logits).item(), torch.mean(critic_value).item()
    
    def get_actor_state_dict(self):
        """
        Returns:
            (dict): state dictionary containing the parameters of the actor network
        """
        return self.actor.state_dict()
    
    def get_critic_state_dict(self):
        """
        Returns:
            (dict): state dictionary containing the parameters of the critic network
        """
        return self.critic.state_dict()