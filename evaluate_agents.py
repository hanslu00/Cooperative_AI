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
import csv

args = parse_args()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
n_observations = 336
n_actions = 4

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

def load_state_dicts_trueskill(agents, indices, state_dicts):
    for agent, i in zip(agents, indices):
        agent.load_state_dict(state_dicts[i])

def initialize_agents(number_of_agents): # teams <- [state_dict1, ..., state_dict20]
    agents = []
    for _ in range(number_of_agents):
        agents.append(Actor(n_observations, n_actions))
    agents[-1] = Actor(n_observations, n_actions+1)
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

def extract_mu_from_csv(csv_file):
    mu_values = []
    with open(csv_file, 'r') as infile:
        reader = csv.reader(infile)
        for row in reader:
            mu_values.append(float(row[0]))
    return mu_values

def extract_std_from_csv(csv_file):
    sigma_values = []
    with open(csv_file, 'r') as infile:
        reader = csv.reader(infile)
        for row in reader:
            sigma_values.append(float(row[0]))
    return sigma_values

def extract_data_from_csvreader(csv_file):
    with open(csv_file, newline='') as csvfile:
        # Create a CSV reader object
        csvreader = csv.reader(csvfile)
        # Read the first row
        first_row = next(csvreader)
        float_list = [float(element) for element in first_row]
    return float_list

def write_list_to_csv(data_list, output_file):
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(data_list)


if __name__ == '__main__':
    saved_state_dicts = ["../new_fcp/frick100/random_agent.pth",
                         "../new_fcp/ppo_versus_random/actor_1.pth",
                         "../new_fcp/ppo_versus_random/actor_4.pth",
                         "../new_fcp/ppo_versus_random/actor_8.pth",
                         "../new_fcp/ppo_versus_random2/actor_2.pth",
                         "../new_fcp/ppo_versus_random2/actor_7.pth",
                         "../new_fcp/ppo_versus_random2/actor_10.pth",
                         "../new_fcp/ppo_versus_random2/actor_17.pth",
                         "../new_fcp/ppo_versus_random_dynamic/actor_1.pth",
                         "../new_fcp/ppo_versus_random_dynamic/actor_2.pth",
                         "../new_fcp/ppo_versus_random_dynamic/actor_6.pth",
                         "../new_fcp/ppo_versus_random_dynamic2/actor_3.pth",
                         "../new_fcp/ppo_versus_random_dynamic2/actor_8.pth",
                         "../new_fcp/ppo_versus_random_dynamic2/actor_12.pth",
                         "../new_fcp/evaluate_ppo_vs_ppo/default_ppo/purple0_actor_5.pth",
                         "../new_fcp/evaluate_ppo_vs_ppo/default_ppo/purple0_actor_7.pth",
                         "../new_fcp/evaluate_ppo_vs_ppo/default2_ppoo/purple0_actor_3.pth",
                         "../new_fcp/evaluate_ppo_vs_ppo/default2_ppoo/purple0_actor_6.pth",
                         "../new_fcp/evaluate_ppo_vs_ppo/dynamic_ppo/purple0_actor_2.pth",
                         "../new_fcp/evaluate_ppo_vs_ppo/dynamic_ppo/purple0_actor_5.pth",
                         "../new_fcp/evaluate_ppo_vs_ppo/dynamic2_ppo/purple0_actor_4.pth",
                         "../new_fcp/evaluate_ppo_vs_ppo/dynamic2_ppo/purple0_actor_7.pth",
                         "../new_fcp/ppo_versus_fcp/default_pro_fcp/FCP_actor_4.pth",
                         "../new_fcp/ppo_versus_fcp/default_pro_fcp/FCP_actor_16.pth",
                         "../new_fcp/ppo_versus_fcp/dynamic_pro_fcp/FCP_actor_8.pth",
                         "../new_fcp/ppo_versus_fcp/dynamic_pro_fcp/FCP_actor_15.pth",
                         "../new_fcp/ppo_versus_fcp/default_fcp/FCP_actor_5.pth",
                         "../new_fcp/ppo_versus_fcp/default_fcp/FCP_actor_9.pth",
                         "../new_fcp/ppo_versus_fcp/dynamic_fcp/FCP_actor_3.pth",
                         "../new_fcp/ppo_versus_fcp/dynamic_fcp/FCP_actor_8.pth",
                         "../fictitious_coplay/first_fcp/FCP_actor_2.pth",
                         "../fictitious_coplay/first_fcp/FCP_actor_5.pth"
                        ]
    
    # agent_nr = 20
    # fcp_nr = 10
    # ppo_nr = 6
    agent_number = 10
    agent_type = "fcp"
    state_dicts = list(map(lambda x: torch.load(x), saved_state_dicts))
    SAVE_PATH = f"dyn_agent{agent_number}/{agent_type}/rating"
    # stds = []

    print("STARTING")
    channel = EngineConfigurationChannel() 
    env = UnityEnvironment(file_name="../Soccer2v2_thingy", seed=args.seed, side_channels=[channel], worker_id = 43) # seed  
    # env = UnityEnvironment(seed=args.seed, side_channels=[channel])
    channel.set_configuration_parameters(width = 8, height = 4, time_scale = 40) # make the unity environment go faster
    # channel.set_configuration_parameters(width = 800, height = 450, time_scale = 1)
    env.reset()

    behavior_names = list(env.behavior_specs.keys())
    team_purple = behavior_names[0] # team 1
    team_blue = behavior_names[1] # team 0


    print(f"Starting with TrueSkill")
    # Have trained N agents, find rating and filter with F 
    agents = initialize_agents(4)
    true_skill_ratings = []
    mu_values = extract_data_from_csvreader("trueskill_pool/final_mu.csv")
    sigma_values = extract_data_from_csvreader("trueskill_pool/final_sigma.csv")

    for i, mu, sig in zip(range(len(state_dicts)), mu_values, sigma_values):
        true_skill_ratings.append(Rating(mu=mu, sigma=sig))

    for char in ["A"]:
        for index in range(21, 31):
            ratings = []
            my_trueskill = Rating()
            agents[3].load_state_dict(torch.load(f"dyn_agent{agent_number}/{agent_type}/actor_{char}{index}.pth"))

            for ep in range(250):
                env.reset() # reset state
                episode_done = False
                reward = 0

                ratings.append(my_trueskill.mu)
                print(my_trueskill)
                # stds.append(extract_stds(true_skill_ratings))
                # shuffle indices
                indices = np.random.choice(np.arange(len(state_dicts)), 3, replace=False)
                print(indices)

                winners = [0, 0] # [blue, purple]
                t1 = [true_skill_ratings[indices[0]], true_skill_ratings[indices[1]]]
                t2 = [true_skill_ratings[indices[2]], my_trueskill]
                load_state_dicts_trueskill(agents, indices, state_dicts)

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
                (_, _), (_, my_trueskill) = rate([t1, t2], ranks=winner)
            
                print("Update true skill episode:", ep)
                # stds_stacked = np.vstack(stds)
                write_list_to_csv(ratings, f'{SAVE_PATH}/{char}{index}.csv')
                # write_list_to_csv(ratings, f'dyn_evaluation3_fcp{index}/seed101.csv')
            
            ratings.append(my_trueskill.mu)
            # stds.append(extract_stds(true_skill_ratings))
            # stds_stacked = np.vstack(stds)
            write_list_to_csv(ratings, f'{SAVE_PATH}/{char}{index}.csv')

    env.close()