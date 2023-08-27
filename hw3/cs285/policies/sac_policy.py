from cs285.policies.MLP_policy import MLPPolicy
import torch
import numpy as np
from cs285.infrastructure import sac_utils
from cs285.infrastructure import pytorch_util as ptu
from torch import nn
from torch import optim
import itertools

torch.autograd.set_detect_anomaly(mode=True)

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

        # QUESTION: What is the formula for this ??? at = exp(Bt) 
        alpha = torch.exp(self.log_alpha) 
        
        
        return alpha


    def get_action(self, obs: np.ndarray, sample=True) -> np.ndarray:
        # TODO: return sample from distribution if sampling
        # if not sampling return the mean of the distribution 

        # additional notes : action should be tensor (action_dim,1) on return 
        # CAREFUL : there can be an error in the action shape, outer loops needs to index into it [0] 

        # detail to make sure obs always remains a float
        

        dist = self.forward(torch.from_numpy(obs).float().to(ptu.device))

        if sample: 
            action = dist.sample()
        else:
            # choose deterministically i.e. return mean of the distribution
            action = dist.mean

        action = ptu.to_numpy(action) 
        return action 
        #return action.numpy()


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

        std = torch.exp(torch.clip(self.logstd,min=min(self.log_std_bounds), max=max(self.log_std_bounds)))
        mean = self.mean_net(observation)
        action_distribution = sac_utils.SquashedNormal(mean,std)

        return action_distribution

    def update(self, obs, critic):
        # TODO Update actor network and entropy regularizer
        # return losses and alpha value


        sampled_action = self.get_action(obs,sample=False) # sample the policy deterministically f(E|St)
        sampled_action_tensor = ptu.from_numpy(sampled_action)
        obs_tensor = ptu.from_numpy(obs)
        dist = self.forward(obs_tensor)

        log_prob_action = dist.log_prob(sampled_action_tensor) # log(pi(st|at))
        

        q1_value, q2_value  = critic.forward(torch.from_numpy(obs).to(ptu.device),sampled_action_tensor)
        q_values_min = torch.min(q1_value,q2_value)

        
        # -1 is for gradient ascent -- NO !! 
        #below is likely incorrect 
        # actor_loss =  -1 * (self.alpha * log_prob_action + (self.alpha * log_prob_action - q_values_min)).mean()
        actor_loss = (self.alpha.detach() * log_prob_action - q_values_min.detach()).mean()
        #actor_loss = -1 * (q_values_min.detach() - self.alpha.detach() * log_prob_action).mean()


        #alpha_loss = (-1 * self.alpha * log_prob_action.detach() - self.alpha * self.target_entropy).mean()

        alpha_loss = (-1 * self.alpha * (log_prob_action.detach() + self.target_entropy)).mean()


        # update the actor
        self.optimizer.zero_grad()
        actor_loss.backward()
        self.optimizer.step()

        # udpate the alpha term

        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        return actor_loss, alpha_loss, self.alpha
            


        

