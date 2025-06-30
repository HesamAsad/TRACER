import copy
import torch
import scipy.stats as stats


class GeneralMovingAverage(object):
    def __init__(self, model, weight_func=None, use_old=False, m_sche_src=0.999, m_sche_tar=0.9999, 
                 m_warm_up=0.1, total_steps=None, ema_up_freq=1):
        """
        General Moving Average class supporting both new Beta-weighted and old momentum-based EMA.
        
        Args:
            model: The model to create moving average for
            weight_func: Weight function for new Beta-weighted approach (ignored if use_old=True)
            use_old: If True, use the old momentum-based EMA implementation
            m_sche_src: Starting momentum value for old implementation
            m_sche_tar: Target momentum value for old implementation  
            m_warm_up: Warm-up fraction of total steps for momentum scheduling
            total_steps: Total training steps (required for old implementation)
            ema_up_freq: Update frequency for old implementation
        """
        self.model = model
        self.use_old = use_old
        
        if use_old:
            # Old implementation parameters
            self.m_sche_src = m_sche_src
            self.m_sche_tar = m_sche_tar
            self.m_warm_up = m_warm_up
            self.total_steps = total_steps
            self.ema_up_freq = ema_up_freq
            self.current_m = m_sche_src
        else:
            # New implementation parameters
            self.weight_func = weight_func
            self.iter = 0
            self.weight = weight_func(self.iter) if weight_func else 1.0
            self.weight_sum = self.weight
        
        # Create moving average model
        self.moving_avg = copy.deepcopy(model)
        for param in self.moving_avg.parameters():
            param.requires_grad = False
        
        # For old implementation: synchronize initial state like the original
        if use_old:
            # This mimics the original: teacher_enc.load_state_dict(model.module.state_dict())
            if hasattr(model, 'module'):
                self.moving_avg.load_state_dict(model.module.state_dict())
            else:
                self.moving_avg.load_state_dict(model.state_dict())

    def update(self, step=None):
        """Update the moving average model"""
        if self.use_old:
            self._update_old(step)
        else:
            self._update_new()
    
    def _update_new(self):
        """New Beta-weighted update implementation"""
        self.iter += 1
        self.weight = self.weight_func(self.iter)
        relative_weight = self.weight / self.weight_sum
        for moving_avg_param, param in zip(self.moving_avg.parameters(), self.model.parameters()):
            moving_avg_param.data = (moving_avg_param + relative_weight * param) / (1 + relative_weight)
        self.weight_sum += self.weight
    
    def _update_old(self, step):
        """Old momentum-based update implementation - exact replica with debug logging"""
        if step is None:
            raise ValueError("Step parameter is required for old EMA implementation")
        
        # Old implementation logic: if ema_up_freq <= 0, do NOTHING (pass)
        if self.ema_up_freq <= 0:
            return  # Do nothing, exactly like the original
        
        # Check conditions
        modulo_condition = (step % self.ema_up_freq) == 0
        final_step_condition = step == self.total_steps
        
        
        # Only update at specified frequency or at the final step
        if modulo_condition or final_step_condition:
            # Calculate momentum with warm-up scheduling
            warm_up_threshold = self.total_steps * self.m_warm_up
            if step < warm_up_threshold:
                self.current_m = (
                    (self.m_sche_tar - self.m_sche_src) 
                    / warm_up_threshold
                ) * step + self.m_sche_src
            else:
                self.current_m = self.m_sche_tar
            
            # Update parameters: param_k = m * param_k + (1-m) * param_q
            # Original used model.module.parameters() for param_q
            model_params = self.model.module.parameters() if hasattr(self.model, 'module') else self.model.parameters()
            for param_q, param_k in zip(model_params, self.moving_avg.parameters()):
                param_k.data.mul_(self.current_m).add_((1 - self.current_m) * param_q.detach().data)

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
    
    @property
    def weight(self):
        """Return current weight/momentum for logging"""
        if self.use_old:
            return self.current_m
        else:
            return self._weight if hasattr(self, '_weight') else 0.0
    
    @weight.setter
    def weight(self, value):
        if not self.use_old:
            self._weight = value


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