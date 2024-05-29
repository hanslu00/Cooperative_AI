from args import parse_args
import numpy as np
import torch
from ppo import PPONet, Deque, Actor
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.base_env import (
    ActionTuple
)
from torch.distributions.categorical import Categorical
from trueskill import Rating, rate
from torch.utils.tensorboard import SummaryWriter
import os
import csv 

args = parse_args()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
n_observations = 336
n_actions = 5

def get_observations(env, team_name):
    observations = np.zeros((2, 0))
    decision_step, _ = env.get_steps(team_name)
    if len(decision_step) > 0:
        for obs in decision_step.obs:
            observations = np.concatenate((observations, obs), axis=1)
    return observations

def get_observations_for_fcp(env, team_name): # agent_id 1
    decision_step, _ = env.get_steps(team_name)
    if len(decision_step) > 0:
        return np.concatenate((decision_step.obs[0][0], decision_step.obs[1][0]))
    
def get_observations_for_partner(env, team_name):
    decision_step, _ = env.get_steps(team_name)
    if len(decision_step) > 0:
        return np.concatenate((decision_step.obs[0][1], decision_step.obs[1][1]))
    
def get_random_action():
    weo = np.random.randint(5)
    return get_actions(weo)

def get_actions(ind):
    actions = [ np.array([[0, 1, 0]]),
                np.array([[0, 2, 0]]),
                np.array([[1, 0, 0]]),
                np.array([[2, 0, 0]]),
                np.array([[0, 0, 0]])]
    return actions[ind]

def set_random_action_for_team(env, team_name):
    decision_step, _ = env.get_steps(team_name)
    if len(decision_step) > 0:
        for agent_id in decision_step.agent_id:
            action = get_random_action()
            env.set_action_for_agent(team_name, agent_id, ActionTuple(discrete=action))
            
def set_action_for_team(env, team_name, team_networks, team_observations):
    actions = np.zeros(2, dtype = np.int64) # actions 0-7
    logprobs = np.zeros(2)
    values = np.zeros(2)

    decision_step, _ = env.get_steps(team_name)

    if len(decision_step) > 0:
        for index, agent_id in enumerate(decision_step.agent_id):
            observation = team_observations[index]
            action_index, logprob, value = team_networks[index].choose_action(observation) # -> size [1]
            action = get_actions(action_index.item())
            env.set_action_for_agent(team_name, agent_id, ActionTuple(discrete=action))

            actions[index] = action_index # store the action chosen
            logprobs[index] = logprob
            values[index] = value

    return actions, logprobs, values

def set_action_for_fcp_agent(env, team_name, fcp_agent, fcp_observations):

    decision_step, _ = env.get_steps(team_name)

    if len(decision_step) > 0:
        action_index, logprob, value = fcp_agent.choose_action(fcp_observations) # -> size [1]
        action = get_actions(action_index.item())
        # env.set_action_for_agent(team_name, 1, ActionTuple(discrete=action))
        env.set_action_for_agent(team_name, 2, ActionTuple(discrete=action))

    return action_index, logprob, value

def set_action_for_partner(env, team_name, partner, partner_obs):
    decision_step, _ = env.get_steps(team_name)

    if len(decision_step) > 0:
        logits = partner(partner_obs)
        dist = Categorical(logits = logits)
        action_index = dist.sample()
        action = get_actions(action_index.item())
        # env.set_action_for_agent(team_name, 3, ActionTuple(discrete=action))
        env.set_action_for_agent(team_name, 3, ActionTuple(discrete=action))

def get_observations_choose_action(env, team_name, team_networks):
    observations = get_observations(env, team_name)
    actions, logprobs, values = set_action_for_team(env, team_name, team_networks, observations)
    return observations, actions, logprobs, values

def add_actor_state_dicts(state_dicts, agents):
    for agent in agents:
        state_dicts.append(agent.get_actor_state_dict())

def add_critic_state_dicts(state_dicts, agents):
    for agent in agents:
        state_dicts.append(agent.get_critic_state_dict())

def initialize_agents(number_of_agents): # teams <- [state_dict1, ..., state_dict20]
    agents = []
    for _ in range(number_of_agents):
        agents.append(Actor(n_observations, n_actions))
    return agents

def set_action_for_opponents(env, team_name, agents, observations):
    decision_step, _ = env.get_steps(team_name)
    if len(decision_step) > 0:
        for index, agent_id in enumerate(decision_step.agent_id):
            observation = observations[index]
            logits = agents[index](observation)
            dist = Categorical(logits = logits)
            action_index = dist.sample()
            action = get_actions(action_index.item())
            env.set_action_for_agent(team_name, agent_id, ActionTuple(discrete=action))

def set_action_from_agents(env, agents, observations):
    ind = 0
    for team_name in env.behavior_specs:
        decision_step, _ = env.get_steps(team_name)
        if len(decision_step) > 0:
            for index, agent_id in enumerate(decision_step.agent_id):
                observation = observations[ind+index]
                logits = agents[ind+index](observation)
                dist = Categorical(logits = logits)
                action_index = dist.sample()
                action = get_actions(action_index.item())
                env.set_action_for_agent(team_name, agent_id, ActionTuple(discrete=action))
        ind += 2

def update_winners(winners, reward):
    if reward > 0: # purple wins
        winners[0] += 1
    else: # draw 
        winners[1] += 1

def get_winner(winners):
    # returns 1 if winners[0] > winners[1]
    if winners[0] > winners[1]:
        return [0, 1]
    elif winners[0] == winners[1]:
        return [1, 1]
    else:
        return [1, 0] 
    
def load_state_dicts_for_team_required_grad_true(team_networks, actor_state_dicts, critic_state_dicts):
    for network, actor_dict, critic_dict in zip(team_networks, actor_state_dicts, critic_state_dicts):
        loaded_actor_dict = torch.load(actor_dict)
        loaded_critic_dict = torch.load(critic_dict)
        network.actor.load_state_dict(loaded_actor_dict)
        network.critic.load_state_dict(loaded_critic_dict)
        for param in network.parameters():
            param.requires_grad = True 

def load_state_dicts_for_team_required_grad_false(team_networks, actor_state_dicts, critic_state_dicts):
    for network, actor_dict, critic_dict in zip(team_networks, actor_state_dicts, critic_state_dicts):
        loaded_actor_dict = torch.load(actor_dict)
        loaded_critic_dict = torch.load(critic_dict)
        network.actor.load_state_dict(loaded_actor_dict)
        network.critic.load_state_dict(loaded_critic_dict)
        for param in network.parameters():
            param.requires_grad = False 

def load_state_dicts_for_trueskill(agents, indices, i, state_dicts):
    for agent, i in zip(agents, indices[i:i+4]):
        agent.load_state_dict(state_dicts[i])

def fcp_load_state_dict(fcp_agent, critic_state_dict, actor_state_dict):
    fcp_agent.actor.load_state_dict(actor_state_dict)
    fcp_agent.critic.load_state_dict(critic_state_dict)
    for param in fcp_agent.parameters():
        param.requires_grad = True

if __name__ == '__main__':
    state_dicts = []
    critic_dicts = []
    agent_nr = 5
    SAVE_PATH = f"dyn_agent{agent_nr}/fcp"
    
    print("STARTING")
    channel = EngineConfigurationChannel() 
    # env = UnityEnvironment(file_name="../Soccer2v2_thingy", seed=args.seed, side_channels=[channel], worker_id = args.worker_id)
    env = UnityEnvironment(file_name="../dynamic_environment/DynamicEnvironment1", seed=args.seed, side_channels=[channel], worker_id = 374)
    # env = UnityEnvironment(seed=args.seed, side_channels=[channel])
    channel.set_configuration_parameters(width = 8, height = 4, time_scale = 35) # make the unity environment go faster
    env.reset()

    behavior_names = list(env.behavior_specs.keys())
    team_purple = behavior_names[0] # team 1
    team_blue = behavior_names[1] # team 0

    # n_episodes = args.n_games
    pro_critic = torch.load(f"dyn_agent{agent_nr}/ppo2/critic_B7.pth")
    pro = torch.load(f"dyn_agent{agent_nr}/ppo2/actor_B7.pth")
    intermediate = torch.load(f"dyn_agent{agent_nr}/ppo1/actor_A10.pth")
    beginner = torch.load(f"dyn_agent{agent_nr}/ppo1/actor_A1.pth")
    
    partners = [beginner, intermediate, pro]
    print("Starting to train FCP_agent that is 'pro'")

    fcp_simsteps = 0
    fcp_agent = PPONet(n_observations, n_actions)
    fcp_load_state_dict(fcp_agent, pro_critic, pro)
    agents = initialize_agents(3) # [opponent0, opponent1, partner]

    n_episodes = 60000
    simulation_steps = 0
    checkpoint = 1

    # statistics from training with PPO 
    fcp_scores = Deque(100)
    fcp_tot_losses = Deque(100)
    fcp_actor_losses = Deque(100)
    fcp_critic_losses = Deque(100)
    fcp_qvals_actor = Deque(100)
    fcp_qvals_critic = Deque(100)

    avg_reward_100 = []
    highest_streak_100 = []
    highest_streak = 0
    current_streak = 0

    for ep in range(1, n_episodes+1):
        env.reset() # reset state
        episode_done = False
        reward = 0
        opponent0, opponent1, partner = np.random.choice(partners, 3)
        agents[0].load_state_dict(opponent0)
        agents[1].load_state_dict(opponent1)
        agents[2].load_state_dict(partner)

        # one training loop 
        while not episode_done:
        # for i in range(10):
            fcp_observations = get_observations_for_fcp(env, team_blue)
            fcp_action, fcp_logprobs, fcp_vals = set_action_for_fcp_agent(env, team_blue, fcp_agent, fcp_observations)

            partner_obs = torch.tensor(get_observations_for_partner(env, team_blue), device=device, dtype=torch.float)
            set_action_for_partner(env, team_blue, agents[2], partner_obs)

            opponents_observations = torch.tensor(get_observations(env, team_purple), device=device, dtype=torch.float)
            set_action_for_opponents(env, team_purple, agents, opponents_observations)
            env.step()
            fcp_simsteps += 1

            d, t = env.get_steps(team_blue)
            if len(t) > 0:
                reward = t.group_reward[0]
                episode_done = True
                print(f"FCP_blue score in episode {ep}: {reward}")

            fcp_agent.store(fcp_observations, fcp_action.item(), fcp_logprobs.item(), fcp_vals.item(), reward, episode_done)

            if fcp_simsteps % args.batch_size == 0:
                fcp_tot_loss, fcp_actor_loss, fcp_critic_loss, fcp_q_actor, fcp_q_critic = fcp_agent.update()

            if episode_done:
                try:
                    fcp_tot_losses.append(fcp_tot_loss)
                    fcp_actor_losses.append(fcp_actor_loss)
                    fcp_critic_losses.append(fcp_critic_loss)
                    fcp_qvals_actor.append(fcp_q_actor)
                    fcp_qvals_critic.append(fcp_q_critic)
                except NameError as e:
                    print("Not defined yet.", e)

                break

        if ep % 2000 == 0:
            print(f"Saving checkpoint {checkpoint} for FCP")
            torch.save(fcp_agent.get_actor_state_dict(), f"{SAVE_PATH}/actor_A{checkpoint}.pth")
            torch.save(fcp_agent.get_critic_state_dict(), f"{SAVE_PATH}/critic_A{checkpoint}.pth")
            checkpoint += 1

        fcp_scores.append(reward)

        if reward > 0:
            current_streak += 1
        else:
            current_streak = 0

        if highest_streak < current_streak:
            highest_streak = current_streak

        if ep % 100 == 0:
            avg_reward_100.append(np.mean(fcp_scores))
            with open(f"{SAVE_PATH}/average_reward_per_100.csv", mode='w', newline='') as file:
                csvwriter = csv.writer(file)
                csvwriter.writerow(avg_reward_100)

        if ep % 100 == 0:
            highest_streak_100.append(highest_streak)
            with open(f"{SAVE_PATH}/highest_streak_per_100.csv", mode='w', newline='') as file:
                csvwriter = csv.writer(file)
                csvwriter.writerow(highest_streak_100)
    
    print("DONE:)")

    env.close()
