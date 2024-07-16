import sys
import gym
import time
import torch
import pybulletgym

from PPO_Actor_Critic import ActorCritic


actorModelPath = "src/AMP/PPO/TrainedModels/6.5h/ppoActor.pth"

env = gym.make('HumanoidPyBulletEnv-v0')
env.render()
obs = env.reset()

obsDims = env.observation_space.shape[0]
actionDims = env.action_space.shape[0]

actorNet = ActorCritic(inputDims = obsDims, outputDims = actionDims)

actorNet.load_state_dict(torch.load(actorModelPath))

while True:
    action = actorNet.forward(obs).detach().numpy()
    obs, reward, done, info = env.step(action)
    time.sleep(0.03)
    
    if done is True:
        env.reset()
    
    env.render()