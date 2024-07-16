import torch
import numpy as np
import torch.nn as nn

from batch import BatchData
from PPO_Actor_Critic import ActorCritic
from torch.optim import Adam as AdamOptimizer
from stats import ModelStatistics
from torch.distributions import MultivariateNormal

class PPO(object):
    def __init__(self,
                 env,
                 saveFreq,
                 savePath,
                 logPath,
                 actorPath = None,
                 criticPath = None,
                 **hyperParams):
        
        """
        Implementation of PPO algorithm based on: https://spinningup.openai.com/en/latest/algorithms/ppo.html#pseudocode
        """
        
        self.__init__hyperParams(hyper_parameters = hyperParams)
        self.env = env
        self.obsDims = self.env.observation_space.shape[0] #44
        self.actionDims = self.env.action_space.shape[0]  #17
        
        if actorPath is not None and criticPath is not None:
            #continue from last checkpoint if available
            self.actor = ActorCritic(input_dimensions=self.observation_dimensions,
                                     output_dimensions=self.action_dimensions)
            self.actor.load_state_dict(torch.load(actorPath))

            self.critic = ActorCritic(input_dimensions=self.observation_dimensions, output_dimensions=1)
            self.critic.load_state_dict(torch.load(criticPath))
        else:
            # initialize actor and critic
            self.actor = ActorCritic(input_dimensions=self.observation_dimensions,
                                     output_dimensions=self.action_dimensions)
            self.critic = ActorCritic(input_dimensions=self.observation_dimensions, output_dimensions=1)

        self.actorOptim = AdamOptimizer(params = self.actor.parameters(), lr = self.lr)
        self.criticOptim = AdamOptimizer(params=self.critic.parameters(), lr = self.lr)
        #create matrix for Diagonal Gaussian Policy
        self.covarianceMat = torch.diag(input=torch.full(size=(self.action_dimensions,), fill_value=0.5), diagonal=0)
        self.saveFreq = saveFreq
        self.savePath = savePath
        self.stats = ModelStatistics()
        self.logPath = logPath
    
    def __init__hyperParams(self, hyperParams):
        """
        Initialize hyper-parameter values.
        https://github.com/araffin/rl-baselines-zoo/blob/master/hyperparams/ppo2.yml
        """
        
        self.lr = 0.005
        self.numBatchTimeSteps = 4800
        self.maxTimeStepsPerEpisode = 1600
        self.numNetUpdatesPerIteration = 5
        self.gamma = 0.95
        self.render = True
        self.seed = None
        
        if self.seed is not None:
            assert(type(self.seed) == int)
            torch.manual_seed(self.seed)
            
        self.normalize = True
        self.clipRange = 0.2
        
        for hyperparam, value in hyperParams.items():
            exec("self." + hyperparam + " = " + str(value))
            
    def getAction(self, obs):

        # query the actor network for a "mean" action (forward propagation)
        meanA = self.actor.forward(obs)
        MGD = MultivariateNormal(loc = meanA, covarianceMatrix = self.covarianceMat)

        #sample an action from the distribution close to the mean
        action = MGD.sample()

        #get log probability for said action
        logProb = MGD.log_prob(value = action)

        return action.detach().numpy(), logProb.detach()

    def rewardsToGo(self, batchRewards):
        RTG = []

        for episodeRewards in reversed(batchRewards):
            #iterate through every reward in the episode
            discountedRewardSum = 0

            for reward in reversed(episodeRewards):
                discountedRewardSum = reward + discountedRewardSum * self.gamma
                RTG.insert(0, discountedRewardSum)

        RTG = torch.tensor(RTG, dtype = torch.float32)
        return RTG
    
    def collectTrajectories(self):

        batchData = BatchData()

        t = 0

        while t < self.numBatchTimeSteps:
            rewardsPerEpisode = []

            obs = self.env.reset()
            isDone = False
            episodeT = None
            
            for episodeT in range(self.maxTimeStepsPerEpisode):
                if self.render:
                    self.env.render()
                
                t += 1
                
                batchData.obss.append(obs)
                
                action, logProb = self.getAction(obs = obs)
                
                obs, reward, isDone, info = self.env.step(action)
                
                rewardsPerEpisode.append(reward)
                batchData.actions.append(action)
                batchData.logProbs.append(logProb)
                
                if isDone:
                    break
                
            batchData.episodeLengths.append(episodeT + 1)
            batchData.rewards.append(rewardsPerEpisode)
        
        batchObservations = torch.tensor(batchData.observations, dtype = torch.float32)
        batchActions = torch.tensor(batchData.actions, dtype = torch.float32)
        batchLogProbs = torch.tensor(batchData.logProbs, dtype = torch.float32)
        batchRewardstoGo = self.rewardsToGo(batchRewards = batchData.rewards)
        
        self.stats.batchEpisodeReturns = batchData.rewards
        self.stats.batchEpisodeLengths = batchData.episodeLengths
        
        return \
            batchObservations, batchActions, batchLogProbs, \
            batchRewardstoGo, batchData.episodeLengths
        
    def calcValueFunction(self, batchObservations, batchActions):
        """
        Estimate the value function for each observation in a batch of observations and logarithmic probabilities of the
        actions taken/executed in that batch with the most recent iteration of the actor network.
        """
        V = self.critic.forward(batchObservations)
        V = V.squeeze()
        
        mu = self.actor.forward(batchObservations)
        
        MGD = MultivariateNormal(loc = mu, covariance_matrix = self.covarianceMat)
        logProb = MGD.log_prob(value = batchActions)
        
        return V, logProb
    
    def normalizeAdvantage(self, advantageFunction, epsilon):
        """
        Standardization of advantage function with epsilon addition.
        """
        return (advantageFunction - advantageFunction.mean()) / (advantageFunction.std() + epsilon)
    
    def calcPiThetaRatio(slef, piTheta, PiThetaK):
        return torch.exp(piTheta = PiThetaK)
    
    def train(self, K):
        """
        Trains the Proximal Policy Optimization model.
        """
        
        k = 0
        i = 0
        
        while k < K:
            batchObservations, batchActions, batchLogProbs, batchRewardstoGo, \
            episodeLengths = self.collectTrajectories()
            
            k += np.sum(episodeLengths)
            
            i += 1
            
            self.stats.elapsedTimeSteps = k
            self.stats.elapsedIterations = i
            
            V, _ = self.calcValueFunction(batchObservations, batchActions)
            advantageFunctionk = batchRewardstoGo - V.detach()
            
            if self.normalize:
                advantageFunctionk = self.normalizeAdvantage(advantageFunction = advantageFunctionk, epsilon = 1e-10)
            
            for _ in range(self.numNetUpdatesPerIteration):
                V, piTheta = self.calcValueFunction(batchObservations = batchObservations, batchActions = batchActions)
                piRatios = self.calcPiThetaRatio(piTheta, batchLogProbs)
                firstTerm = piRatios * advantageFunctionk
                
                secondTerm = torch.clamp(piRatios, 1 - self.clipRange, 1 + self.clipRange) * advantageFunctionk
                
                ppoClipObj = (-torch.min(firstTerm, secondTerm)).mean()
                self.actorOptim.zero_grad()
                ppoClipObj.backward(retain_graph = True)
                self.actorOptim.step()
                
                regressionMSE = nn.MSELoss()(V, batchRewardstoGo)
                self.criticOptim.zero_grad()
                regressionMSE.backward()
                self.criticOptim.step()
                
                self.stats.actorLosses.append(ppoClipObj.detach())
            
            log = self.stats.__str__()
            f = open(self.logPath + "/log.txt", "a")
            f.write(log)
            f.close()
            
            if i % self.saveFreq == 0:
                torch.save(self.actor.state_dict(), self.savePath + "/ppoActor.pth")
                torch.save(self.critic.state_dict(), self.savePath + "/ppoActor.pth")
                
                                   
            
                
