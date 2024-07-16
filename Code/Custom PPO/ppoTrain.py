import gym
import pybulletgym

from alg import PPO
from os.path import exists

def trainModel(env,
          hyperparams,
          totalTimeSteps,
          saveFreq,
          savePath,
          actorModel,
          criticModel,
          logPath):
    """
    Function for training the model for a specified gym environment.
    """
    print("Training in progress...\n")
    
    if not exists(path=actorModel) and not exists(path=criticModel):
        print("Training model for the first time (actor and critic models don't exist).")
        
    model = PPO(env = env,
                saveFreq = saveFreq,
                savePath = savePath,
                logPath = logPath,
                actorPath = actorModel,
                criticPath = criticModel,
                **hyperparams)
    
    model.train(K = totalTimeSteps)
    
    if __name__ == "__main__":
        env = gym.make('HumanoidPyBulletEnv-v0')
        
        hyperparams = {
            "lr" : 2.5e-4,
            "numBatchTimeSteps": 4800,
            "maxTimeStepsPerEpisode": 1600,
            "numNetUpdatesPerIteration": 20,
            "gamma": 0.99,
            "render": True,
            "seed": None,
            "normalize": True,
            "clipRange": 0.2 
        }  
        
        trainModel(env = env,
                   hyperparams = hyperparams,
                   totalTimeSteps = 1e12,
                   saveFreq = 5,
                   savePath = "src\AMP\PPO\TrainedModels"
                   ) 