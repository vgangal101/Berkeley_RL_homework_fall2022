from collections import OrderedDict

from cs285.critics.bootstrapped_continuous_critic import \
    BootstrappedContinuousCritic
from cs285.infrastructure.replay_buffer import ReplayBuffer
from cs285.infrastructure.utils import *
from cs285.policies.MLP_policy import MLPPolicyAC
from .base_agent import BaseAgent
import gym
from cs285.policies.sac_policy import MLPPolicySAC
from cs285.critics.sac_critic import SACCritic
import cs285.infrastructure.pytorch_util as ptu
from cs285.infrastructure.sac_utils import soft_update_params
import torch # mod added during implementation.

class SACAgent(BaseAgent):
    def __init__(self, env: gym.Env, agent_params):
        super(SACAgent, self).__init__()

        self.env = env
        self.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]
        self.agent_params = agent_params
        self.gamma = self.agent_params['gamma']
        self.critic_tau = 0.005
        self.learning_rate = self.agent_params['learning_rate']

        self.actor = MLPPolicySAC(
            self.agent_params['ac_dim'],
            self.agent_params['ob_dim'],
            self.agent_params['n_layers'],
            self.agent_params['size'],
            self.agent_params['discrete'],
            self.agent_params['learning_rate'],
            action_range=self.action_range,
            init_temperature=self.agent_params['init_temperature']
        )
        self.actor_update_frequency = self.agent_params['actor_update_frequency']
        self.critic_target_update_frequency = self.agent_params['critic_target_update_frequency']

        self.critic = SACCritic(self.agent_params)
        self.critic_target = copy.deepcopy(self.critic).to(ptu.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.training_step = 0
        self.replay_buffer = ReplayBuffer(max_size=100000)

    def update_critic(self, ob_no, ac_na, next_ob_no, re_n, terminal_n):
        # TODO: 
        # 1. Compute the target Q value. 
        # HINT: You need to use the entropy term (alpha)
        

        obs_tensor = ptu.from_numpy(ob_no)
        action_tensor = torch.from_numpy(ac_na).float().to(ptu.device)
        


        next_action_sampled = self.actor.get_action(ob_no)
        next_action_sampled_tensor = ptu.from_numpy(next_action_sampled)
        next_obs_tensor = ptu.from_numpy(next_ob_no)


        target_q1_vals, target_q2_vals = self.critic_target(next_obs_tensor,next_action_sampled_tensor)
        min_target_q_val = torch.min(target_q1_vals,target_q2_vals)


        # implementation todo - compute the learning target for the critic 
        rewards_tensor = ptu.from_numpy(re_n)
        terminals_tensor = ptu.from_numpy(terminal_n)       

        # compute the log_probs for all actions 
        #batch_size = next_obs_tensor.shape[0]

        # all_log_prob_action = []
        # for i in range(batch_size):
        #     dist = self.actor.forward(next_obs_tensor[i])
        #     action_log_prob = dist.log_prob(next_action_sampled_tensor[i])
        #     all_log_prob_action.append(action_log_prob)
        
        # log_probs = torch.stack(all_log_prob_action)

        dist = self.actor.forward(next_obs_tensor)
        log_probs = dist.log_prob(next_action_sampled_tensor)
  
        terminals_tensor = terminals_tensor.unsqueeze(-1)
        rewards_tensor = rewards_tensor.unsqueeze(-1)


        learning_target = rewards_tensor + self.gamma * (1 - terminals_tensor) * (min_target_q_val.detach() - self.actor.alpha.detach() * log_probs.detach())


        # 2. Get current Q estimates and calculate critic loss
       
        q1_values, q2_values = self.critic.forward(obs_tensor,action_tensor)
        q1_loss = self.critic.loss(q1_values,learning_target)
        q2_loss = self.critic.loss(q2_values,learning_target)
        
        critic_loss = q1_loss + q2_loss

        # 3. Optimize the critic
        self.critic.optimizer.zero_grad()  
        critic_loss.backward()
        self.critic.optimizer.step()
        
        return critic_loss

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
        # TODO 
        # 1. Implement the following pseudocode:
        # for agent_params['num_critic_updates_per_agent_update'] steps,
        #     update the critic

        critic_loss = None

        for _ in range(self.agent_params['num_critic_updates_per_agent_update']):
            #def update_critic(self, ob_no, ac_na, next_ob_no, re_n, terminal_n):
            loss_critic = self.update_critic(ob_no, ac_na, next_ob_no, re_n, terminal_n)
            
        critic_loss = loss_critic.item()
        
        # 2. Softly update the target every critic_target_update_frequency (HINT: look at sac_utils)
        if self.training_step % self.critic_target_update_frequency == 0:
            soft_update_params(self.critic,self.critic_target,self.critic_tau)


        # 3. Implement following pseudocode:
        # If you need to update actor
        # for agent_params['num_actor_updates_per_agent_update'] steps,
        #     update the actor

        actor_loss = None 
        alpha_loss = None 
        alpha = None 
        
        for _ in range(self.agent_params['num_actor_updates_per_agent_update']):
            _actor_loss, _alpha_loss, _alpha  = self.actor.update(ob_no,self.critic)
            
        actor_loss = _actor_loss.item()
        alpha_loss = _alpha_loss.item()
        alpha = _alpha.item()
        
        # IMPLEM detail: are all the quantities going to be torch tensors or numpy ndarray ??
        # do this after verifying above components


        # 4. gather losses for logging
        loss = OrderedDict()
        loss['Critic_Loss'] = critic_loss
        loss['Actor_Loss'] = actor_loss
        loss['Alpha_Loss'] =   alpha_loss
        loss['Temperature'] =  alpha

        return [loss]

    def add_to_replay_buffer(self, paths):
        self.replay_buffer.add_rollouts(paths)

    def sample(self, batch_size):
        return self.replay_buffer.sample_random_data(batch_size)
