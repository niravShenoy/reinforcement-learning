#!/usr/bin/env python3

import os
import argparse
import json
import time
import random
import logging
import re
import random

import numpy as np

from IPython.display import clear_output
import matplotlib.pyplot as plt
# %matplotlib inline

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torchsummary import summary

from env import Env
from ppo import PolicyNetwork

from torch.utils.tensorboard import SummaryWriter

curr_path = os.path.abspath(os.path.join(os.getcwd()))


# To train in the order of tasks (For imitation learning)

def sortFiles(files):
    def convert(text): return int(text) if text.isdigit() else text
    def alphanum_key(key): return [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(files, key=alphanum_key)


def parseDataset(args):
    dataset_dir = os.path.abspath(os.path.join(os.getcwd(), 'datasets'))
    dataset_type = os.path.join(dataset_dir, args.dataset)
    if args.validation:
        data = os.path.join(dataset_type, 'val')
    else:
        data = os.path.join(dataset_type, 'train')

    data_tasks = os.path.join(data, 'task')
    data_seq = os.path.join(data, 'seq')

    data_taskfiles = sortFiles([f for f in os.listdir(
        data_tasks) if os.path.isfile(os.path.join(data_tasks, f))])
    data_seqfiles = sortFiles([f for f in os.listdir(
        data_seq) if os.path.isfile(os.path.join(data_seq, f))])
    
    return data_tasks, data_seq, data_taskfiles, data_seqfiles


def getTask(data_tasks, taskfile):
    task = json.load(open(os.path.join(data_tasks, taskfile)))

    # Parse task -> Can convert to a function
    pre_x = task['pregrid_agent_row']
    pre_y = task['pregrid_agent_col']
    pre_dir = task['pregrid_agent_dir']

    post_x = task['postgrid_agent_row']
    post_y = task['postgrid_agent_col']
    post_dir = task['postgrid_agent_dir']

    walls = task['walls']
    pregrid_markers = task['pregrid_markers']
    postgrid_markers = task['postgrid_markers']

    height = task['gridsz_num_rows']
    width = task['gridsz_num_cols']
    return Env(pre_x, pre_y, pre_dir, post_x, post_y, post_dir, walls, pregrid_markers, postgrid_markers, height, width)

def getSeq(data_seq, taskfile):
    task = json.load(open(os.path.join(data_seq, taskfile)))
    expert_actions = task['sequence']
    return expert_actions


# Using Generalized Advantage Estimation (GAE) to calculate batch advantage
def computeAdvantage(latest_value, rewards, masks, values, done, gamma=0.99, lambda_=0.95):
    # extended_values = torch.cat([values, torch.tensor(next_value).reshape((-1,1))], dim=0)
    gae = 0
    advantages = torch.zeros_like(rewards)

    # Calculating in reverse to apply discounting
    for step in reversed(range(len(rewards))):
        if step == len(rewards) - 1:
            nextnonterminal = 1.0 - done
            next_value = latest_value
        else:
            nextnonterminal = masks[step + 1]
            next_value = values[step + 1]
            delta = rewards[step] + gamma * values[step + 1] * nextnonterminal - values[step]
            gae = delta + gamma * lambda_ * nextnonterminal * gae
            advantages[step] = gae

    # Returning in a compatible shape
    return advantages

def test_agent(args, agent, env, device, evaluate = False):
    state = env.reset()
    step_limit = args.H * 2
    if args.render:
        env.render()

    terminated = False
    truncated = False
    total_reward = 0
    total_steps = 0

    for _ in range(step_limit):
        state = torch.FloatTensor(state).to(device)
        with torch.no_grad():
            action, action_log_prob, entropy, value, _ = agent(
                torch.reshape(state, (-1,)))
        next_state, reward, truncated, terminated = env.step(
            action.cpu().numpy())

        state = next_state
        total_reward += reward
        total_steps += 1

        if terminated or truncated:
            break

        if args.render:
            env.render()

    return total_reward, total_steps, terminated

def save_checkpoint(checkpoint, filepath):
    torch.save(checkpoint, filepath)

def evaluateModel(args, agent, device, data_tasks, data_taskfiles, data_seq, data_seqfiles):
    total_completed = 0.0
    total_minimal_steps = 0.0
    total_reward = 0.0

    for i in range(len(data_taskfiles)):
        env = getTask(data_tasks, data_taskfiles[i])
        state = env.reset()
        f = open("data_easy.txt", "a")
        f.write("Initial State:\n")
        f.close()
        env.render()
        f = open("data_easy.txt", "a")
        f.write("Goal State:\n")
        f.close()
        env.render(True)
        expert_actions = getSeq(data_seq, data_seqfiles[i])
        reward, steps, completed = test_agent(args, agent, env, device)
        # print("Validation Reward: ", reward, "Steps: ", steps, "Completed: ", completed)
        if completed:
            total_completed += 1.0
        if len(expert_actions) >= steps and completed:
            total_minimal_steps += 1.0
        total_reward += reward

    percent_completed = total_completed / len(data_taskfiles)*100
    percent_minimal_steps = total_minimal_steps / len(data_taskfiles)*100
    avg_reward = total_reward / len(data_taskfiles)
    return percent_completed, percent_minimal_steps, avg_reward

def behavioral_reward(policy_action, expert_action):
    reward = 0
    if policy_action == expert_action:
        reward += 1
    else:
        rewards -= 1
    return reward

def bc_train(agent, env, expert_actions, device, num_epochs=1, clip_epsilon=0.2):
    optimizer = optim.Adam(agent.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    action_index = {"putMarker" : 0, 
                    "turnLeft" : 1,
                    "turnRight": 2, 
                    "pickMarker": 3, 
                    "move" : 4, 
                    "finish": 5
                    }

    for epoch in range(num_epochs):

        states = torch.zeros((len(expert_actions), args.num_envs) +
                                env.observation_space.shape).to(device)
        actions = torch.zeros((len(expert_actions), args.num_envs) +
                                env.action_space.shape).to(device)
        logprobs = torch.zeros(len(expert_actions)).to(device)
        rewards = torch.zeros(len(expert_actions)).to(device)
        values = torch.zeros(len(expert_actions)).to(device)
        masks = torch.zeros(len(expert_actions)).to(device)

        state = env.reset()

        for expert_action in expert_actions:
            # Sampling Actions using the Policy
            expert_vector = torch.zeros(6)
            expert_action_ind = torch.tensor(action_index[expert_action])
            expert_vector[expert_action_ind] = 1
            state = torch.FloatTensor(state).to(device)
            action, action_log_prob, entropy, value, action_logits= agent(
                torch.reshape(state, (-1,)))

            loss = criterion(action_logits.reshape(1,-1), expert_action_ind.reshape(1,))

            next_state, reward, truncated, terminated = env.step(expert_action_ind.cpu().numpy())
            state = next_state

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return agent


def main(args):
    run_name = f"{args.env_name}__{args.seed}__{int(time.time())}"
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] - %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S', filename=os.path.join(curr_path, args.dataset, 'logs', run_name + '.log'))
    logging.info('%s; Run Name = %s; LR = %s; Rollout Length = %s; Epochs = %s; Minibatch Size = %s; Validation = %s',
                 args.dataset, run_name, args.learning_rate, args.H, args.num_epochs, args.mini_batch_size, args.validation)
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    agent = PolicyNetwork(320, 6, 512).to(device)
    # summary(agent, (96,))
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate)
    data_train_tasks, data_train_seq, data_train_taskfiles, data_train_seqfiles = parseDataset(
        args)

    init_path = os.path.join(curr_path, args.dataset, 'model', 'init_model_' + str(args.dataset) + '.pt')
    torch.save(agent.state_dict(), init_path)

    best_path = os.path.join(curr_path, args.dataset, 'model', 'best_model_' + str(args.dataset) + '.pt')
    latest_path = os.path.join(curr_path, args.dataset, 'model', 'latest_model_' + str(args.dataset) + '.pt')

    best_loss = torch.tensor(float('inf'))

    # Initialize Dummy environment
    env = getTask(data_train_tasks, data_train_taskfiles[0])

    ########### Training Loop #############

    # Initial Agent does policy rollouts
    # This generates rollout data which is then used to train the policy
    # The policy is then updated and the process is repeated

    # args.H -> Number of steps to rollouts
    # args.num_envs -> Number of environments to rollouts (in case of vectorized envs)
    # (H * num_envs) -> Number of data points for training
    ########################################

    # Initializing variables to store details of the training

    global_step = 0
    start_time = time.time()
    training_rewards = []
    training_loss = []


    # Number of updates to be performed
    num_updates = args.total_timesteps // args.batch_size

    f = open("data_easy.txt", "w")
    f.write("Training Begins\n")
    f.close()

    task_ids = np.arange(len(data_train_taskfiles))
    np.random.shuffle(task_ids)

    # If Validation, load model and perform validation
    if args.validation:

        data_val_tasks, data_val_seq, data_val_taskfiles, data_val_seqfiles = parseDataset(
        args)
        for path in [best_path, latest_path]:
            checkpoint = torch.load(path)
            agent = agent.to(device)
            agent.load_state_dict(checkpoint['state_dict'])
            percent_completed, percent_minimal_steps, avg_reward = evaluateModel(args, agent, device, data_val_tasks, data_val_taskfiles, data_val_seq, data_val_seqfiles)
            model_type = 'BEST'
            if path == latest_path:
                model_type = 'LATEST'
            print("Percent Completed: ", percent_completed, "Percent Minimal Steps: ", percent_minimal_steps, "Average Reward: ", avg_reward)
            logging.info("Model: %s - Percent Completed: %s, Percent Minimal Steps: %s, Average Reward: %s", model_type, percent_completed, percent_minimal_steps, avg_reward)
        return



    for itr in range(1, num_updates + 1):

        state = env.reset()

        # # Pretraining the agent with the expert data
        for i in range(args.num_episodes):
            task_id = task_ids[(itr-1) * args.num_episodes + i]
            env = getTask(data_train_tasks, data_train_taskfiles[task_id])
            expert_actions = getSeq(data_train_seq, data_train_seqfiles[task_id])
            agent = bc_train(agent, env, expert_actions, device)


        states = torch.zeros((args.num_episodes * args.H, args.num_envs) +
                             env.observation_space.shape).to(device)
        actions = torch.zeros((args.num_episodes * args.H, args.num_envs) +
                              env.action_space.shape).to(device)
        logprobs = torch.zeros((args.num_episodes * args.H, args.num_envs)).to(device)
        rewards = torch.zeros((args.num_episodes * args.H, args.num_envs)).to(device)
        values = torch.zeros((args.num_episodes * args.H, args.num_envs)).to(device)
        masks = torch.zeros((args.num_episodes * args.H, args.num_envs)).to(device)
        

        # Running the agent with the current policy

        for epi in range(0, args.num_episodes):
            task_id = task_ids[(itr-1) * args.num_episodes + i]
            env = getTask(data_train_tasks, data_train_taskfiles[task_id])

            for step in range(0, args.H):
                global_step += 1 * args.num_envs
                state = torch.FloatTensor(state).to(device)

                # Sampling Actions using the Policy
                with torch.no_grad():
                    action, action_log_prob, entropy, value, _ = agent(
                        torch.reshape(state, (-1,)))
                    values[step] = value

                logprobs[step] = action_log_prob

                # Perform action in the environment
                next_state, reward, truncated, terminated = env.step(
                    action.cpu().numpy())
                rewards[step] = reward
                masks[step] = 1 - (terminated or truncated)

                states[step] = state
                state = next_state
                actions[step] = action

                if terminated or truncated:
                    state = env.reset()

                # Check progess every 1000 steps
                if global_step % 1000 == 0:
                    total_reward = 0.0
                    for _ in range(10):
                        curr_reward ,_ ,_ = test_agent(args, agent, env, device)
                        total_reward += curr_reward
                    avg_reward = total_reward / 10
                    training_rewards.append(avg_reward)
                    print("Step= {}, Avg Reward= {}".format(
                        global_step, avg_reward))
                    logging.info("Step= {}, Avg Reward= {}".format(global_step, avg_reward))

        next_state = torch.FloatTensor(next_state).to(device)
        with torch.no_grad():
            _, _, _, value, _ = agent(torch.reshape(next_state, (-1,)))
            advantages = computeAdvantage(value, rewards, masks, values, terminated or truncated)

        exp_returns = advantages + values

        # Perform PPO Update
        mini_batch_size = args.mini_batch_size
        num_epochs = args.num_epochs

        training_steps = 0

        # Reshaping the data to be compatible for batch training
        batch_states = states.reshape((-1,) + env.observation_space.shape).to(device)
        batch_actions = actions.reshape((-1,) + env.action_space.shape).to(device)
        batch_logprobs = logprobs.reshape(-1).to(device)
        batch_exp_returns = exp_returns.reshape(-1).to(device)
        batch_advantages = advantages.reshape(-1).to(device)


        ind = np.arange(args.batch_size)

        for epoch in range(num_epochs):
            np.random.shuffle(ind)
            for i in range(0, args.batch_size, mini_batch_size):
                begin_index = i
                end_index = i + mini_batch_size
                sampled_ind = ind[begin_index:end_index]

                training_steps += 1
                minibatch_advantages = batch_advantages[sampled_ind]

                # for state, action, logprobs_old, exp_return, advantage in zip(batch_states, batch_actions, batch_logprobs, batch_exp_returns, batch_advantages):
                next_action, logprobs_new, entropy, value_new, _ = agent(batch_states[sampled_ind].reshape((mini_batch_size,-1)))

                # Ratio of the new log prob to the old log prob
                policy_ratio = torch.exp(logprobs_new - batch_logprobs[sampled_ind])
                clipped_policy_ratio = torch.clamp(policy_ratio, 1 - args.clip_epsilon, 
                                1 + args.clip_epsilon)

                # Normalize the advantages
                minibatch_advantages = (minibatch_advantages - minibatch_advantages.mean()) / (minibatch_advantages.std() + 1e-8)

                # Actor Loss updates the policy to pick actions that maximize the expected return
                actor_loss = -torch.min(policy_ratio * minibatch_advantages,
                                clipped_policy_ratio * minibatch_advantages).mean()

                # Critic Loss updates the baseline to be closer to the expected return
                critic_loss = 0.5 * (batch_exp_returns[sampled_ind] - value_new).pow(2).mean()

                # The final Objective Function combines the Actor and Critic Losses and also to increase entropy to encourage exploration
                # Entropy is the measure of how unpredictable the policy is
                # c1 = 0.5 and c2 = 0.01
                final_loss = actor_loss + (args.c1 * critic_loss) + (args.c2 * entropy.mean())
                final_loss.to(device)
                
                if final_loss.item() < best_loss.item():
                    best_loss = final_loss
                    checkpoint = {
                        'epoch': epoch+1,
                        'training_loss': best_loss.item(),
                        'state_dict': agent.state_dict(),
                    }
                    save_checkpoint(checkpoint, best_path)

                optimizer.zero_grad()
                final_loss.backward()
                optimizer.step()

                if epoch == (num_epochs - 1) and end_index == args.batch_size:
                    print("Training Loss: ", final_loss.item(), ", Actor Loss: ", actor_loss.item(), ", Critic Loss:", critic_loss.item())
                    training_loss.append(final_loss.item())
                    checkpoint = {
                        'epoch': epoch+1,
                        'training_loss': final_loss.item(),
                        'state_dict': agent.state_dict(),
                    }
                    save_checkpoint(checkpoint, latest_path)

    data = {'loss': training_loss, 'reward': training_rewards}
    with open(os.path.join(curr_path, args.dataset, 'logs', run_name + '.json'), 'w') as f:
        json.dump(data, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch PPO Implementation')

    parser.add_argument('--env-name', default='gridworld-v0',
                        help='name of the environment to run')

    parser.add_argument('--run_id', type=int, default=0,
                        help='run id (default: 0)')

    parser.add_argument('--seed', type=int, default=543,
                        help='random seed (default: 543)')

    parser.add_argument('--total-timesteps', type=int, default=256000,
                        help='number of timesteps (default: 512000)')

    parser.add_argument('--render', default=False, 
                        help='render environment (default: False)')

    parser.add_argument('--dataset', type=str, default='data_easy',
                        help='dataset to train from (default: data_easy, data_medium, data)')

    parser.add_argument('--cuda', action='store_true',
                        default=True, help='run on CUDA (default: True)')

    ####### Hyperparameters #######

    # lr = 1e-5
    # gamma = 0.99
    # Total Timesteps = 512000 -> data_easy; 1536000 -> data_medium; 3072000 -> data
    # args.H = 128
    # args.num_envs = 1
    # episodes = 50 -> data_easy; 150 -> data_medium; 300 -> data
    # mini_batch_size = 80 
    # num_updates = 80
    # clip_epsilon = 0.2
    # c1 = 0.5
    # c2 = 0.01
    # num_epochs = 3 per update
    ################################

    parser.add_argument('--num-envs', type=int, default=1,
                        help='number of parallel environments (default: 1)')

    parser.add_argument('--H', type=int, default=64,
                        help='number of steps per rollout (default: 64)')

    parser.add_argument('--learning-rate', type=float,
                        default=1e-3, help='learning rate (default: 1e-5)')

    parser.add_argument('--gamma', type=float, default=0.99,
                        help='discount factor for rewards (default: 0.99)')

    parser.add_argument('--clip-epsilon', type=float, default=0.2,
                        help='discount factor for rewards (default: 0.2)')

    parser.add_argument('--c1', type=float, default=0.5,
                        help='Value Coefficient')

    parser.add_argument('--c2', type=float, default=0.0,
                        help='Entropy Coefficient')

    parser.add_argument("--num-minibatches", type=int, default=80,
                        help="the number of mini-batches")
    
    parser.add_argument("--num-episodes", type=int, default=50,
                        help="the number of episodes")

    parser.add_argument("--num-epochs", type=int, default=3, 
                        help="the number of epochs")

    parser.add_argument("--validation", type=bool, default=False, 
                        help="Validating the model")

    # parser.add_argument("--test", type=bool, default=False, 
    #                     help="Testing the model")

    args = parser.parse_args()
    args.batch_size = args.num_envs * args.num_episodes * args.H
    args.mini_batch_size = int(args.batch_size // args.num_minibatches)

    main(args)
