import numpy as np

class GaussianWhiteNoiseProcess():
    def __init__(self, size, mu=0., sigma=1.):
        self.size = size
        self.mu = mu
        self.sigma = sigma

    def sample(self):
        sample = np.random.normal(self.mu, self.sigma, self.size)
        return sample
    
    def reset(self):
        pass

class OUNoise:
    def __init__(self, size, theta=0.15, mu=0.0, sigma=0.3):
        self.size = size
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.state = None
        self.reset()

    def reset(self):
        self.state = np.ones(self.size) * self.mu

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state