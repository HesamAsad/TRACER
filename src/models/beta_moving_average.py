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


class ExponentialMovingAverage(object):
    def __init__(self, model, get_momentum_fn, update_frequency: int = 1):
        """
        EMA teacher wrapper.

        Args:
            model: student model to track (will be deep-copied for teacher)
            get_momentum_fn: callable(step:int)->float returning momentum in [0,1)
            update_frequency: update every N optimization steps (>=1)
        """
        self.model = model
        self.get_momentum_fn = get_momentum_fn
        self.iter = 0  # tracks global steps passed in, if provided
        self.update_frequency = max(1, int(update_frequency))  # kept for backward compatibility
        self.momentum = get_momentum_fn(self.iter)
        self.moving_avg = copy.deepcopy(model)
        for param in self.moving_avg.parameters():
            param.requires_grad = False

    def update(self, global_step: int = None):
        """
        Update EMA weights.

        Args:
            global_step: optional global training step used for momentum scheduling.
        """
        # Use provided global_step for scheduling; fall back to internal counter
        if global_step is None:
            self.iter += 1
            step_for_sched = self.iter
        else:
            self.iter = global_step
            step_for_sched = global_step

        # Compute momentum from schedule at the global step
        self.momentum = self.get_momentum_fn(step_for_sched)
        m = self.momentum
        for moving_avg_param, param in zip(self.moving_avg.parameters(), self.model.parameters()):
            moving_avg_param.data = moving_avg_param.data * m + param.data * (1.0 - m)

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
        if hasattr(self.moving_avg, 'save'):
            self.moving_avg.save(path)
        else:
            torch.save(self.moving_avg.state_dict(), path)

    @property
    def module(self):
        return self.moving_avg.module if hasattr(self.moving_avg, 'module') else self.moving_avg


def create_linear_warmup_ema_momentum(src_momentum: float, tar_momentum: float, warmup_ratio: float, total_iterations: int):
    """
    Create a momentum scheduler for EMA with linear ramp from src to tar during warmup portion.

    - For the first warmup_ratio of total iterations, momentum increases linearly from src_momentum to tar_momentum.
    - After warmup, it stays at tar_momentum.

    Args:
        src_momentum: starting momentum (e.g., 0.05)
        tar_momentum: target/final momentum (e.g., 0.9)
        warmup_ratio: fraction of total iterations used for linear warmup (e.g., 0.2)
        total_iterations: total number of optimization steps
    Returns:
        fn(step)->momentum
    """
    warmup_steps = float(total_iterations) * float(warmup_ratio)

    def get_momentum(step: int) -> float:
        # Match deprecated logic: linear increase for first (total_iterations * warmup_ratio) steps
        if warmup_steps <= 0.0:
            return tar_momentum
        if float(step) < warmup_steps:
            t = float(step) / warmup_steps
            return src_momentum + t * (tar_momentum - src_momentum)
        else:
            return tar_momentum

    return get_momentum