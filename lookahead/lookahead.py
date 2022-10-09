import torch
import math 

class Lookahead(torch.optim.Optimizer):
    def __init__(self, optimizer, alpha=0.5, pullback_every=3, pullback_momentum="reset"):
        '''
        :param optimizer:inner optimizer
        :param alpha(float): linear interpolation factor. 1.0 recovers the inner optimizer.
        :param pullback_momentum (str): change to inner optimizer momentum on interpolation update
        '''
        from collections import defaultdict 
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f'Invalid slow update rate: {alpha}')
        self.optimizer = optimizer
        self.param_groups = self.optimizer.param_groups
        self.alpha = alpha
        self.step_counter = 0
        self.pullback_every = pullback_every
        assert pullback_momentum in ["reset", "pullback", "none"]
        self.pullback_momentum = pullback_momentum
        self.state = defaultdict(dict)
        for group in self.optimizer.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                param_state['cached_params'] = torch.zeros_like(p.data)
                param_state['cached_params'].copy_(p.data)

    def __getstate__(self):
        return {
            'state': self.state,
            'optimizer': self.optimizer,
            'alpha': self.alpha,
            'step_counter': self.step_counter,
            'k':self.k,
            'pullback_momentum': self.pullback_momentum
        }

    def zero_grad(self):
        self.optimizer.zero_grad()
        
    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    def _backup_and_load_cache(self):
        for group in self.optimizer.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                param_state['backup_params'] = torch.zeros_like(p.data)
                param_state['backup_params'].copy_(p.data)
                p.data.copy_(param_state['cached_params'])

    def _clear_and_load_backup(self):
        for group in self.optimizer.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                p.data.copy_(param_state['backup_params'])
                del param_state['backup_params']

    def step( self, closure=None):
        self.step_counter += 1
        s = self.optimizer.step(closure)
        if self.step_counter == self.pullback_every:
          self.step_counter = 0
          self.pullback()
        return s

    def pullback(self):
      for group in self.optimizer.param_groups:
          for p in group['params']:
              param_state = self.state[p]
              p.data.mul_(self.alpha).add_(param_state['cached_params'], alpha=1.0-self.alpha)
              param_state['cached_params'].copy_(p.data)
              if self.pullback_momentum == "pullback":
                  internal_momentum = self.optimizer.state[p]["momentum_buffer"]
                  self.optimizer.state[p]["momentum_buffer"] = internal_momentum.mul_(self.alpha).add_(param_state["cached_mom"], alpha=1.0-self.alpha)
                  param_state["cached_mom"] = self.optimizer.state[p]["momentum_buffer"]
              elif self.pullback_momentum == "reset":
                  self.optimizer.state[p]["momentum_buffer"] = torch.zeros_like(p.data)
