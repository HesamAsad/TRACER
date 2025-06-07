import copy
import torch
import scipy.stats as stats


class GeneralMovingAverage(object):
    def __init__(self, model, weight_func):
        self.model = model
        self.weight_func = weight_func
        self.iter = 0
        self.weight = weight_func(self.iter)
        self.weight_sum = self.weight
        self.moving_avg = copy.deepcopy(model)
        for param in self.moving_avg.parameters():
            param.requires_grad = False

    def update(self):
        self.iter += 1
        self.weight = self.weight_func(self.iter)
        relative_weight = self.weight / self.weight_sum
        for moving_avg_param, param in zip(self.moving_avg.parameters(), self.model.parameters()):
            moving_avg_param.data = (moving_avg_param + relative_weight * param) / (1 + relative_weight)
        self.weight_sum += self.weight

    def __call__(self, x: torch.Tensor):
        return self.moving_avg(x)

    def train(self, mode=True):
        self.moving_avg.train(mode)

    def eval(self):
        self.train(False)

    def state_dict(self):
        return self.moving_avg.state_dict()

    def load_state_dict(self, state_dict):
        self.moving_avg.load_state_dict(state_dict)

    def save(self, path):
        """Save the moving average model"""
        if hasattr(self.moving_avg, 'save'):
            self.moving_avg.save(path)
        else:
            torch.save(self.moving_avg.state_dict(), path)

    @property
    def module(self):
        return self.moving_avg.module if hasattr(self.moving_avg, 'module') else self.moving_avg


def create_beta_weight_function(beta_param, total_iterations):
    """
    Create a weight function based on Beta distribution
    
    Args:
        beta_param: Beta parameter (used for both alpha and beta of Beta distribution)
        total_iterations: Total number of iterations in training
    
    Returns:
        weight_func: Function that takes iteration and returns weight
    """
    beta_dist = stats.beta(beta_param, beta_param)
    
    def weight_func(iteration):
        # Normalize iteration to [0, 1] range
        normalized_iter = (iteration + 0.5) / (total_iterations + 1)
        return beta_dist.pdf(normalized_iter)
    
    return weight_func 