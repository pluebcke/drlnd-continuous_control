from collections import deque
import numpy as np
import torch

import time

from unityagents import UnityEnvironment
from agent_td3 import T3DAgent
from tools import save_results

env_path = './data/Reacher_Linux_NoVis_MP/Reacher.x86_64'
save_path = './results/'


actor_settings = dict(layer1_size=400, layer2_size=300, out_noise=3e-3)
critic_settings = dict(layer1_size=400, layer2_size=300, out_noise=3e-3)

settings = {
    'batch_size': 64,            # Number of experience samples per training step
    'buffer_size': int(3e6),     # Max number of samples in the replay memory
    'gamma': 0.99,               # Reward decay factor
    'tau': 1e-3,                 # Update rate for the slow update of the target networks
    'lr_actor': 5e-4,            # Actor learning rate
    'lr_critic': 5e-4,           # Critic learning rate
    'action_noise': 0.4,         # Noise added during episodes played
    'action_clip': 1.0,          # Actions are clipped to +/- action_clip
    'target_action_noise': 0.4,  # Noise added during the critic update step
    'target_noise_clip': 0.2,    # Noise clip for the critic update step
    'number_steps': 1,           # Number of steps for roll-out, currently not used
    'optimize_critic_every': 2,  # Update the critic only every X update steps
    'pretrain_steps': int(10000),# Number of random actions played before training starts
    'actor_settings': actor_settings,
    'critic_settings': critic_settings}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
# Start the envirnment and the agent
env = UnityEnvironment(file_name=env_path, seed=2)
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
agent = T3DAgent(env, brain, brain_name, device, settings)

# Setting up various lists and queues for logging results
rewards = []
c_a_loss = []
c_b_loss = []
a_loss = []
avg_rewards = []
reward_mean = deque(maxlen=100)
episode = 0
step = 0

actor_loss_queue = deque(maxlen=10)
critic_a_loss_queue = deque(maxlen=10)
critic_b_loss_queue = deque(maxlen=10)

# Take random steps before training starts
agent.pretrain()

#Start training the aent
startTime = time.time()
while True:
    # Taking a step in the environment and save results in the ReplayBuffer
    reward, std_reward = agent.take_step()
    step = step + 1

    if step % 20 == 0:
        for _ in range(10):
            # Improve the policies
            actor_loss, critic_a_loss, critic_b_loss = agent.learn()
            # Log results
            actor_loss_queue.append(actor_loss)
            critic_a_loss_queue.append(critic_a_loss)
            critic_b_loss_queue.append(critic_b_loss)
        c_a_loss.append(np.mean(critic_a_loss_queue))
        c_b_loss.append(np.mean(critic_b_loss_queue))

    if reward != - 1:
        # If the episode is over log results and print status messages
        reward_mean.append(reward)
        rewards.append(reward)
        a_loss.append(np.mean(actor_loss_queue))

        avg_rewards.append(np.mean(reward_mean))
        episodeTime = time.time() - startTime
        startTime = time.time()
        print("\rEps: " + str(episode) + " rew: " + str(reward) + " std: " + str(std_reward) + " AvgRew:" + str(
            np.mean(reward_mean)) + " crcloss: " + str(np.mean(critic_a_loss_queue)) + " " + str(np.mean(critic_b_loss_queue)) + " actor loss: " + str(np.mean(actor_loss_queue)) + " tim: " + str(episodeTime), end="")
        if episode % 100 == 0:
            print("")
        episode += 1

        if episode == 1000 or np.mean(reward_mean) > 40:
            break

save_results(save_path, agent, settings, rewards, avg_rewards, a_loss, c_a_loss, c_b_loss)



