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

def get_observations_in_tensor(env, team_name):
    observations = np.zeros((2, 0))
    decision_step, _ = env.get_steps(team_name)
    if len(decision_step) > 0:
        for obs in decision_step.obs:
            observations = np.concatenate((observations, obs), axis=1)
    return torch.tensor(observations, device=device, dtype=torch.float)

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



def get_observations_choose_action(env, team_name, team_networks):
    observations = get_observations(env, team_name)
    actions, logprobs, values = set_action_for_team(env, team_name, team_networks, observations)
    return observations, actions, logprobs, values

def set_action_from_actor(env, team_name, actor, team_observations):
    decision_step, _ = env.get_steps(team_name)

    if len(decision_step) > 0:
        for index, agent_id in enumerate(decision_step.agent_id):
            observation = team_observations[index]
            logits = actor(observation)
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

def initialize_dict(teams): # teams <- [state_dict1, ..., state_dict20]
    dct = {}
    for i, state_dict in enumerate(teams):
        dct[i+1] = {"state_dict":state_dict, "rating":1500}
    return dct 

def actor_load_cp(actor, cp):
    cp = torch.load(cp)
    actor.load_state_dict(cp)

def update_winners(winners, reward):
    if reward > 0: # purple wins
        winners[0] += 1
    else: # blue wins
        winners[1] += 1

def get_winner(winners):
    # returns 1 if winners[0] > winners[1]
    if winners[0] > winners[1]:
        return [0, 1] # purple wins
    elif winners[0] == winners[1]:
        return [1, 1]
    else:
        return [1, 0] # blue wins
    
def load_state_dicts_for_team(team_networks, actor_state_dict, critic_state_dict):
    for network in team_networks:
        network.actor.load_state_dict(actor_state_dict)
        network.critic.load_state_dict(critic_state_dict)
        for param in network.parameters():
            param.requires_grad = True
    
def load_state_dicts_for_trueskill(agents, indices, i, state_dicts):
    for agent, i in zip(agents, indices[i:i+4]):
        agent.load_state_dict(state_dicts[i])

def initialize_agents(number_of_agents): # teams <- [state_dict1, ..., state_dict20]
    agents = []
    for _ in range(number_of_agents):
        agents.append(Actor(n_observations, n_actions))
    return agents

def extract_rating(true_skill_ratings):
    lst = []
    for rating in true_skill_ratings:
        lst.append(rating.mu)
    return lst

def extract_stds(true_skill_ratings):
    lst = []
    for rating in true_skill_ratings:
        lst.append(rating.sigma)
    return lst


if __name__ == '__main__':
    folder = "dyn_agent5/ppo2"
    saved_state_dicts = [f"{folder}/actor_A1.pth",
                         f"{folder}/actor_A2.pth",
                         f"{folder}/actor_A3.pth",
                         f"{folder}/actor_A4.pth",
                         f"{folder}/actor_A5.pth",
                         f"{folder}/actor_A6.pth",
                         f"{folder}/actor_A7.pth",
                         f"{folder}/actor_A8.pth",
                         f"{folder}/actor_A9.pth",
                         f"{folder}/actor_A10.pth",
                         f"{folder}/actor_B1.pth",
                         f"{folder}/actor_B2.pth",
                         f"{folder}/actor_B3.pth",
                         f"{folder}/actor_B4.pth",
                         f"{folder}/actor_B5.pth",
                         f"{folder}/actor_B6.pth",
                         f"{folder}/actor_B7.pth",
                         f"{folder}/actor_B8.pth",
                         f"{folder}/actor_B9.pth",
                         f"{folder}/actor_B10.pth"]

    state_dicts = list(map(lambda x: torch.load(x), saved_state_dicts))
    SAVE_PATH = f"{folder}/group_rating"
    ratings = []
    stds = []

    print("STARTING")
    channel = EngineConfigurationChannel() 
    env = UnityEnvironment(file_name="../Soccer2v2_thingy", seed=args.seed, side_channels=[channel], worker_id = 29) # seed  
    # env = UnityEnvironment(seed=args.seed, side_channels=[channel])
    channel.set_configuration_parameters(width = 80, height = 45, time_scale = 50) # make the unity environment go faster
    # channel.set_configuration_parameters(width = 800, height = 450, time_scale = 1)
    env.reset()

    behavior_names = list(env.behavior_specs.keys())
    team_purple = behavior_names[0] # team 1
    team_blue = behavior_names[1] # team 0


    print("Starting with TrueSkill")
    # Have trained N agents, find rating and filter with F 
    agents = initialize_agents(4)
    true_skill_ratings = []

    for i in range(len(state_dicts)):
        true_skill_ratings.append(Rating())

    for ep in range(250):
        env.reset() # reset state
        episode_done = False
        reward = 0

        ratings.append(extract_rating(true_skill_ratings))
        stds.append(extract_stds(true_skill_ratings))
        # shuffle indices
        indices = np.arange(len(state_dicts)) # divisible by 4 
        np.random.shuffle(indices)
        print(indices)

        for i in range(0, len(indices), 4):
            winners = [0, 0] # [blue, purple]
            t1 = [true_skill_ratings[indices[i]], true_skill_ratings[indices[i+1]]]
            t2 = [true_skill_ratings[indices[i+2]], true_skill_ratings[indices[i+3]]]
            load_state_dicts_for_trueskill(agents, indices, i, state_dicts)

            # 2 v 2 game in 3000*5 environment steps 
            # for _ in range(3000):
            for _ in range(1200):
                b_observations = torch.tensor(get_observations(env, team_blue), device=device, dtype=torch.float)
                p_observations = torch.tensor(get_observations(env, team_purple), device=device, dtype=torch.float)
                all_observations = torch.cat((p_observations, b_observations), dim=0)
                set_action_from_agents(env, agents, all_observations)
                env.step()

                d, t = env.get_steps(team_purple)
                if len(t) > 0:
                    reward = t.group_reward[0]
                    update_winners(winners, reward)
                    env.reset()
        
            print("Puple", winners, "Blue", "         Episode:", ep)
            ### Check who wins and update ratings
            winner = get_winner(winners)
            (true_skill_ratings[indices[i]], true_skill_ratings[indices[i+1]]), \
            (true_skill_ratings[indices[i+2]], true_skill_ratings[indices[i+3]]) = rate([t1, t2], ranks=winner)
    
        print("Update true skill episode:", ep)
        stds_stacked = np.vstack(stds)
        ratings_stacked = np.vstack(ratings)

        with open(f'{SAVE_PATH}/agents_mu.csv', 'w') as file:
            # Iterate over each column index
            for col_index in range(ratings_stacked.shape[1]):
                # Get the column from the array
                column = ratings_stacked[:, col_index]
                # Convert the column to a string and write it to the file
                file.write(','.join(map(str, column)) + '\n')

        with open(f'{SAVE_PATH}/agents_sigma.csv', 'w') as file:
            # Iterate over each column index
            for col_index in range(stds_stacked.shape[1]):
                # Get the column from the array
                column = stds_stacked[:, col_index]
                # Convert the column to a string and write it to the file
                file.write(','.join(map(str, column)) + '\n')

    
    ratings.append(extract_rating(true_skill_ratings))
    stds.append(extract_stds(true_skill_ratings))
    stds_stacked = np.vstack(stds)
    ratings_stacked = np.vstack(ratings)


    # Write each column to a text file
    with open(f'{SAVE_PATH}/agents_mu.csv', 'w') as file:
        # Iterate over each column index
        for col_index in range(ratings_stacked.shape[1]):
            # Get the column from the array
            column = ratings_stacked[:, col_index]
            # Convert the column to a string and write it to the file
            file.write(','.join(map(str, column)) + '\n')

    with open(f'{SAVE_PATH}/agents_sigma.csv', 'w') as file:
        # Iterate over each column index
        for col_index in range(stds_stacked.shape[1]):
            # Get the column from the array
            column = stds_stacked[:, col_index]
            # Convert the column to a string and write it to the file
            file.write(','.join(map(str, column)) + '\n')

    env.close()
