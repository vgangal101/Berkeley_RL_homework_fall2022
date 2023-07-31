from .base_critic import BaseCritic
from torch import nn
from torch import optim
import numpy as np
from cs285.infrastructure import pytorch_util as ptu
from cs285.infrastructure import sac_utils
import torch

class SACCritic(nn.Module, BaseCritic):
    """
        Notes on notation:

        Prefixes and suffixes:
        ob - observation
        ac - action
        _no - this tensor should have shape (batch self.size /n/, observation dim)
        _na - this tensor should have shape (batch self.size /n/, action dim)
        _n  - this tensor should have shape (batch self.size /n/)

        Note: batch self.size /n/ is defined at runtime.
        is None
    """
    def __init__(self, hparams):
        super(SACCritic, self).__init__()
        self.ob_dim = hparams['ob_dim']
        self.ac_dim = hparams['ac_dim']
        self.discrete = hparams['discrete']
        self.size = hparams['size']
        self.n_layers = hparams['n_layers']
        self.learning_rate = hparams['learning_rate']

        # critic parameters
        self.gamma = hparams['gamma']
        self.Q1 = ptu.build_mlp(
            self.ob_dim + self.ac_dim,
            1,
            n_layers=self.n_layers,
            size=self.size,
            activation='relu'
        )
        self.Q2 = ptu.build_mlp(
            self.ob_dim + self.ac_dim,
            1,
            n_layers=self.n_layers,
            size=self.size,
            activation='relu'
        )
        self.Q1.to(ptu.device)
        self.Q2.to(ptu.device)
        self.loss = nn.MSELoss()

        self.optimizer = optim.Adam(
            self.parameters(),
            self.learning_rate,
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor):
        # TODO: return the two q values
        
        # Implem detail:
         # values is a dict , containing keys q1_value, and q2_value. 
        values = {}

        q1_value = self.Q1(obs).gather(1,action)
        q2_value = self.Q2(obs).gather(1,action)

        values['q1_value'] = q1_value
        values['q2_value'] = q2_value

        return values



        