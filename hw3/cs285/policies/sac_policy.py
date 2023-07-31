from cs285.policies.MLP_policy import MLPPolicy
import torch
import numpy as np
from cs285.infrastructure import sac_utils
from cs285.infrastructure import pytorch_util as ptu
from torch import nn
from torch import optim
import itertools

class MLPPolicySAC(MLPPolicy):
    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 discrete=False,
                 learning_rate=3e-4,
                 training=True,
                 log_std_bounds=[-20,2],
                 action_range=[-1,1],
                 init_temperature=1.0,
                 **kwargs
                 ):
        super(MLPPolicySAC, self).__init__(ac_dim, ob_dim, n_layers, size, discrete, learning_rate, training, **kwargs)
        self.log_std_bounds = log_std_bounds
        self.action_range = action_range
        self.init_temperature = init_temperature
        self.learning_rate = learning_rate

        self.log_alpha = torch.tensor(np.log(self.init_temperature)).to(ptu.device)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.learning_rate)

        self.target_entropy = -ac_dim

    @property
    def alpha(self):
        # TODO: Formulate entropy term
        
        # original instructions ( above) say to formulate the entropy term , but that does not make sense 
        
        alpha = torch.exp(self.log_alpha)
        #return entropy
        return alpha


    def get_action(self, obs: np.ndarray, sample=True) -> np.ndarray:
        # TODO: return sample from distribution if sampling
        # if not sampling return the mean of the distribution 

        # additional notes : action should be tensor (action_dim,1) on return 
        # CAREFUL : there can be an error in the action shape, outer loops needs to index into it [0] 

        dist = self.forward(torch.from_numpy(obs.to(ptu.device)))

        if sample: 
            action = dist.sample()
        else:
            # choose deterministically i.e. return mean of the distribution
            action = dist.mean
        return action.numpy()

    # This function defines the forward pass of the network.
    # You can return anything you want, but you should be able to differentiate
    # through it. For example, you can return a torch.FloatTensor. You can also
    # return more flexible objects, such as a
    # `torch.distributions.Distribution` object. It's up to you!
    def forward(self, observation: torch.FloatTensor):
        # TODO: Implement pass through network, computing logprobs and apply correction for Tanh squashing

        # HINT: 
        # You will need to clip log values
        # You will need SquashedNormal from sac_utils file 

        # Implementation notes: 
        # send back the action distribution i.e. a torch.distributions.Distribution object 

        std = torch.exp(torch.clip(self.agent.log_std,min=min(self.log_std_bounds), max=max(self.log_std_bounds)))
        mean = self.mean_net(observation)
        action_distribution = sac_utils.SquashedNormal(mean,std)

        return action_distribution

    def update(self, obs, critic):
        # TODO Update actor network and entropy regularizer
        # return losses and alpha value

        # Implem notes: 
        # critic will give you the 2 Q-values involved 
        # make 


        #return actor_loss, alpha_loss, self.alpha
        # adjusting the return statement here 
        
        # logic
        

        # for this part do we need to redo the computation where we sample from a spherical gaussian ? do we need to reproduce the contents of forward ?? 
        
        input_tensor = torch.from_numpy(obs).to(ptu.device)

        dist = self.forward(input_tensor)
        action = dist.mean

        q_values = critic.forward(input_tensor,action)
        q_values_min = torch.min(q_values['q1_value'],q_values['q2_value'])
        





        


        return actor_loss, entropy_loss, self.alpha

