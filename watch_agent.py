import time
import torch
from unityagents import UnityEnvironment
from agent_td3 import T3DAgent

env_path = './data/Reacher_Linux_MP/Reacher.x86_64'
actor_strategy_path = "./results/run003/_actor_net.pt"
critic_strategy_path = "./results/run003/_critic_net.pt"
number_episodes = 20
max_time = 999

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
    'optimize_critic_every': 2,  # Update the critic only every X update steps
    'pretrain_steps': int(10000),# Number of random actions played before training starts
    'actor_settings': actor_settings,
    'critic_settings': critic_settings}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
env = UnityEnvironment(file_name=env_path, seed=2)
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
agent = T3DAgent(env, brain, brain_name, device, settings)

# Uncomment to watch a dumb agent
#for _ in range(number_episodes):
#    for _ in range(max_time):
#        reward, _ = agent.take_step()
#        time.sleep(0.003)

agent.load_nets(actor_strategy_path, critic_strategy_path)
agent.set_action_noise(0.0)

for _ in range(number_episodes):
    for _ in range(max_time):
        reward, _ = agent.take_step(train_mode=False)
        time.sleep(0.001)

env.close()