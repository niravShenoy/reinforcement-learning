#!/usr/bin/env python3

import os
import json
import argparse
import re
import torch

from env import Env
from ppo import PolicyNetwork

curr_path = os.path.abspath(os.path.join(os.getcwd()))

def sortFiles(files):
    def convert(text): return int(text) if text.isdigit() else text
    def alphanum_key(key): return [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(files, key=alphanum_key)

def parseDataset(data_tasks):
    data_taskfiles = sortFiles([f for f in os.listdir(
        data_tasks) if os.path.isfile(os.path.join(data_tasks, f))])
    return data_taskfiles

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

def testAgent(args, agent, env, device, run_name, evaluate = False):
    state = env.reset()
    step_limit = args.episode_length
    if args.render:
        env.render()

    action_index = ["putMarker", "turnLeft", "turnRight", "pickMarker", "move", "finish"]

    terminated = False
    truncated = False
    total_reward = 0
    total_steps = 0
    seq = []

    for _ in range(step_limit):
        state = torch.FloatTensor(state).to(device)
        with torch.no_grad():
            action, action_log_prob, entropy, value, _ = agent(
                torch.reshape(state, (-1,)))
        seq.append(action_index[action.item()])
        next_state, reward, truncated, terminated = env.step(
            action.cpu().numpy())

        state = next_state
        total_reward += reward
        total_steps += 1

        if terminated or truncated:
            break

        if args.render:
            env.render()

    data = {'sequence': seq}
    with open(os.path.join(curr_path, 'seq', run_name + '_seq.json'), 'w') as f:
        json.dump(data, f)
    return total_reward, total_steps, terminated

def evaluateModel(args, agent, device, data_tasks, data_taskfiles):
    total_completed = 0.0
    total_minimal_steps = 0.0
    total_reward = 0.0

    f = open("data_easy.txt", "w")
    f.write("Starting Testing\n")
    f.close()

    for i in range(len(data_taskfiles)):
        env = getTask(data_tasks, data_taskfiles[i])
        run_name = data_taskfiles[i].split('_')[0]
        state = env.reset()
        if args.render:
            f = open("data_easy.txt", "a")
            f.write("Initial State:\n")
            f.close()
            env.render()
            f = open("data_easy.txt", "a")
            f.write("Goal State:\n")
            f.close()
            env.render(True)
        reward, steps, completed = testAgent(args, agent, env, device, run_name)
        print("Validation Reward: ", reward, "Steps: ", steps, "Completed: ", completed)
        if completed:
            total_completed += 1.0
        total_reward += reward

    percent_completed = total_completed / len(data_taskfiles)*100
    percent_minimal_steps = total_minimal_steps / len(data_taskfiles)*100
    avg_reward = total_reward / len(data_taskfiles)
    return percent_completed, percent_minimal_steps, avg_reward

def main(args):
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")


    torch.manual_seed(args.seed)

    modelname = args.model + '.pt'
    model_path = os.path.join(curr_path, modelname)
    checkpoint = torch.load(model_path)

    agent = PolicyNetwork(96, 6, 256).to(device)
    agent.load_state_dict(checkpoint['state_dict'])

    # for param_tensor in checkpoint['state_dict']:
    #     print(param_tensor, "\t", checkpoint['state_dict'][param_tensor].size())

    if args.path != '':
        dataset_path = args.path
    else:
        dataset_path = os.path.join(curr_path, 'datasets', args.dataset, 'test_without_seq', 'task')
        if not os.path.exists(dataset_path):
            print("Dataset not found")
            return

    data_taskfiles = parseDataset(dataset_path)


    percent_completed, percent_minimal_steps, avg_reward = evaluateModel(args, agent, device, dataset_path, data_taskfiles[:100])
    print("Percent Completed: ", percent_completed, "Percent Minimal Steps: ", percent_minimal_steps, "Average Reward: ", avg_reward)
    


    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test PPO Algorithm')

    parser.add_argument('--path', type=str, default='', help='Path to the dataset')
    parser.add_argument('--model', type=str, default='latest_model', help='model to use')
    parser.add_argument('--dataset', type=str, default='data', help='dataset to use')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--cuda', action='store_true', default=True, help='use cuda')
    parser.add_argument('--render', default=True, help='render the environment')
    parser.add_argument('--episode-length', type=int, default=64, help='maximum episode length')

    args = parser.parse_args()
    main(args)


