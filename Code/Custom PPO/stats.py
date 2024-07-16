import numpy as np

class ModelStatistics(object):
    def __init__(self):
        self.elapsedTimeSteps = 0
        self.elapsedIterations = 0
        self.batchEpisodeLengths = []
        self.batchEpisodeReturns = []
        self.actorLosses = []
        
    def __str__(self):
        str = ""
        
        averageEpisodeLengths = round(np.mean(self.batchEpisodeLengths), 3)
        averageEpisodeReturns = round(np.mean([np.sum(episodicReward) for episodicReward in self.batchEpisodeReturns]), 3)
        averageActorLoss = np.mean([losses.float().mean() for losses in self.actorLosses])
        
        print("Iteration: {}".format(self.elapsedIterations))
        print("Elapsed time-steps: {}".format(self.elapsedTimeSteps))
        print("Average Episodic Length: {}".format(averageEpisodeLengths))
        print("Average Episodic Returns: {}".format(averageEpisodeReturns))
        print("Average actor loss: {}\n".format(averageActorLoss))

        str += "Iteration: {}\n".format(self.elapsedIterations)
        str += "Elapsed time-steps: {}\n".format(self.elapsedTimeSteps)
        str += "Average Episodic Length: {}\n".format(averageEpisodeLengths)
        str += "Average Episodic Returns: {}\n".format(averageEpisodeReturns)
        str += "Average actor loss: {}\n\n".format(averageActorLoss)
        
        self.batchEpisodeLengths = []
        self.batchEpisodeReturns = []
        self.actorLosses = []
        
        return str