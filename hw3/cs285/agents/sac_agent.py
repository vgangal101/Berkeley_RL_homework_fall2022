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
        
        min_target_q_func_vals = []
        log_prob_actions = []
        batch_size = ob_no.shape[0]

        # !!!!! - YOU NEED TO RETHINK, REWRITE AND REDO THIS PART USING BATCH OPERATIONS !!!
        for i in range(batch_size): 
            #ob = ptu.from_numpy(ob_no[i]).to(ptu.device)
            ob = ob_no[i]
            
            action = self.actor.get_action(ob)
            #action = torch.from_numpy(action).to(ptu.device) # is this the correct action to be using ????
            #action = ptu.from_numpy(ac_na[i],ptu.device)

            #min_q_val = torch.min(self.critic_target(ob,action))
            q_vals = self.critic_target(torch.from_numpy(ob).to(ptu.device),torch.from_numpy(action).to(ptu.device))
            q1_val = q_vals['q1_value']
            q2_val = q_vals['q2_value']
            min_target_q_val = torch.min(q1_val,q2_val)
            min_target_q_func_vals.append(min_target_q_val)

            dist = self.actor.forward(torch.from_numpy(ob).to(ptu.device))
            log_prob_action = dist.log_prob(torch.from_numpy(action).to(ptu.device))
            log_prob_actions.append(log_prob_action)


        #all_obs = ptu.from_numpy(ob_no,ptu.device)
        #dists = self.actor.forward(all_obs)        
        min_target_q_func_vals = torch.cat(min_target_q_func_vals)
        log_prob_actions = torch.cat(log_prob_actions)
        #all_rewards = ptu.from_numpy(re_n,ptu.device)
        all_rewards = torch.from_numpy(re_n).to(ptu.device)
        terminal_n_tensor = torch.from_numpy(terminal_n).to(ptu.device)

        q_func_target_val = all_rewards + self.gamma * (1-terminal_n_tensor) * (min_target_q_func_vals - self.actor.alpha * log_prob_actions)


        # 2. Get current Q estimates and calculate critic loss
        q_est_func1 = []
        q_est_func2 = []
        for i in range(batch_size):
            ob = ptu.from_numpy(ob_no[i]).to(ptu.device)
            
            #action = ptu.from_numpy(ac_na[i],ptu.device)
            action = ptu.from_numpy(ac_na[i]).to(ptu.device)
            q_values = self.critic(ob,action)
            q_est_func1.append(q_values['q1_value'])
            q_est_func2.append(q_values['q2_value'])

        q_est_func1 = torch.cat(q_est_func1)
        q_est_func2 = torch.cat(q_est_func2)

        q1_loss = self.critic.loss(q_est_func1,q_func_target_val)
        q2_loss = self.critic.loss(q_est_func2,q_func_target_val)
        
        critic_loss = q1_loss + q2_loss

        # 3. Optimize the critic  
        critic_loss.backward()
        self.critic.optimizer.step()
        
        return critic_loss

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
        # TODO 
        # 1. Implement the following pseudocode:
        # for agent_params['num_critic_updates_per_agent_update'] steps,
        #     update the critic

        all_critic_loss = []

        for _ in range(self.agent_params['num_critic_updates_per_agent_update']):
            #def update_critic(self, ob_no, ac_na, next_ob_no, re_n, terminal_n):
            loss_critic = self.update_critic(ob_no, ac_na, next_ob_no, re_n, terminal_n)
            all_critic_loss.append(loss_critic)

        #critic_loss = torch.cat(all_critic_loss)
        critic_loss = torch.tensor(all_critic_loss)

        # 2. Softly update the target every critic_target_update_frequency (HINT: look at sac_utils)
        if self.training_step % self.critic_target_update_frequency == 0:
            soft_update_params(self.critic,self.critic_target,self.critic_tau)


        # 3. Implement following pseudocode:
        # If you need to update actor
        # for agent_params['num_actor_updates_per_agent_update'] steps,
        #     update the actor

        all_actor_loss = []
        all_alpha_loss = []
        temperature_values = []
        for _ in range(self.agent_params['num_actor_updates_per_agent_update']):
            loss_actor, loss_alpha, alpha  = self.actor.update(ob_no,self.critic)
            all_actor_loss.append(loss_actor)
            all_alpha_loss.append(loss_alpha)
            temperature_values.append(alpha)
        
        actor_loss = torch.tensor(all_actor_loss)
        alpha_loss = torch.tensor(all_alpha_loss)
        temperature_values = torch.tensor(temperature_values)
        # IMPLEM detail: are all the quantities going to be torch tensors or numpy ndarray ??
        # do this after verifying above components


        # 4. gather losses for logging
        loss = OrderedDict()
        loss['Critic_Loss'] = ptu.to_numpy(critic_loss)
        loss['Actor_Loss'] = ptu.to_numpy(actor_loss)
        loss['Alpha_Loss'] = ptu.to_numpy(alpha_loss)  
        loss['Temperature'] =  ptu.to_numpy(temperature_values)

        return [loss]

    def add_to_replay_buffer(self, paths):
        self.replay_buffer.add_rollouts(paths)

    def sample(self, batch_size):
        return self.replay_buffer.sample_random_data(batch_size)
