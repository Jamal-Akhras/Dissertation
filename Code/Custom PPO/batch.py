class BatchData(object):
    def __init__(self):
        self.observations = []
        self.actions = []
        self.logProbs = []
        self.rewards = []
        self.rewardsToGo = []
        self.epsiodeLengths = []