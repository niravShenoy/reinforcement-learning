# Gridworld for any task

The Gridworld problem has been proven to solved for a given task in a limited state space using Exact Solution methods. These Exact Solution Methods however, are task specific and need to be calculated from scratch for every new task that it is presented with. So it can be safe to say that these methods will not scale with larger MDPs to solve multiple tasks. To generalize the Gridworld problem to any task, we apply On-Policy methods that aim to find the optimal policy that can transform the pre-grid to the post-grid.

Policies can be optimized using Policy Gradients, which aim to shift policies based on the rewards it receives. A neural network is used to take the state encoding as input and give the action as output based on the policy. The goal of the policy optimization is to maximize the probability of the trajectories that return the highest rewards.

In this project, the Proximal Policy Optimization (PPO) method is applied to the set of training tasks with the goal of learning the optimal policy to transform the pre-grid to the post-grid. The key concept of this technique is to provide a more stable optimization for high order MDPs by making sure the policy updates don't diverge beyond a certain range while at the same time improving the value estimation with every iteration.

## About the code

Files containing the code:
To run the program
```
$ python train.py --args
```

train_bc.py - This contains the LATEST CODE for running the PPO Algorithm.  It Implements Behavioral Cloning to make the policy learn from expert demonstrations. It also has the code for validation

train.py - Contains the code to read the data and run the policy on the environment and update the policy using Proximal Policy Optimization <br />
ppo.py - Neural Network having Actor and Critic heads with the Actor giving the output of the action and the critic estimating the value given the current state of the environment <br/>
env.py - Contains the code to implement the environment for the Gridworld

## Code Layout

Initial Agent does policy rollouts
This generates rollout data which is then used to train the policy
The policy is then updated and the process is repeated

args.H -> Number of steps to rollouts
args.num_envs -> Number of environments to rollouts (in case of vectorized envs)
(H * num_envs) -> Number of data points for training

## Hyperparameters

lr = 1e-5 <br/>
gamma = 0.99 <br/>
Total Timesteps = 512000 -> data_easy; 1536000 -> data_medium; 3072000 -> data <br/>
args.H = 128 <br/>
args.num_envs = 1 <br/>
episodes = 50 -> data_easy; 150 -> data_medium; 300 -> data <br/>
mini_batch_size = 80  <br/>
num_updates = 80 <br/>
clip_epsilon = 0.2 <br/>
c1 = 0.5 <br/>
c2 = 0.01 <br/>
num_epochs = 3 per update
