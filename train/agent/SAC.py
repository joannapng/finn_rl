import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from config import cfg
import pdb
import os.path as osp
import math


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1 or classname.find("Linear") != -1:
        # m.weight.data.normal_(0.0, 0.1)
        nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
        if m.bias is not None:
            m.bias.data.fill_(0)

def soft_update_from_to(source, target, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )

# _rec means recurrent version
class SoftActorCritic(nn.Module):
    def __init__(self):
        super(SoftActorCritic, self).__init__()
        self.hidden_state_size = cfg.DIM_FIRST_LAYER

        self.actor = SoftActor([cfg.DST_S_LEN,], cfg.A_DIM, self.hidden_state_size, cfg.NUM_POLICY)
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=cfg.ACTOR_LR_RATE)

        self.critic_1 = SoftCritic([cfg.DST_S_LEN,], cfg.A_DIM, self.hidden_state_size)
        self.critic_opt_1 = optim.Adam(self.critic_1.parameters(), lr=cfg.CRITIC_LR_RATE)

        self.critic_2 = SoftCritic([cfg.DST_S_LEN,], cfg.A_DIM, self.hidden_state_size)
        self.critic_opt_2 = optim.Adam(self.critic_2.parameters(), lr=cfg.CRITIC_LR_RATE)

        self.dist = torch.distributions.Categorical
        self.target_critic_1 = SoftCritic([cfg.DST_S_LEN,], cfg.A_DIM, self.hidden_state_size)
        self.target_critic_2 = SoftCritic([cfg.DST_S_LEN,], cfg.A_DIM, self.hidden_state_size)
        
        self.target_critic_1.eval()
        self.target_critic_2.eval()
        if cfg.EMODE == "pretrain":
            if cfg.SCALARIZATION_METHOD == 1:
                self.weights = torch.from_numpy(cfg.MO_WEIGHTS).float()
            elif cfg.SCALARIZATION_METHOD == 2:
                self.ths = torch.from_numpy(cfg.MO_THS).float()
            else:
                raise NotImplementedError
            if cfg.AUTOMATIC_ENTROPY_TUNING:
                self.alpha = torch.ones(cfg.NUM_POLICY, 1)
                self.alpha[:] = cfg.ENTROPY_WEIGHT
                self.alpha.requires_grad = True
                self.alpha_optim = optim.Adam([self.alpha], lr=cfg.ALPHA_LR_RATE)
            else:
                self.alpha = torch.ones(cfg.NUM_POLICY, 1)
                self.alpha[:] = cfg.ENTROPY_WEIGHT

        if cfg.NUM_POLICY > 1:
            if cfg.MULTI_POLICY_MODE == 0: # all data
                self.multi_policy_masks = torch.ones(cfg.BATCH_SIZE, cfg.NUM_AGENTS, cfg.NUM_POLICY)
            elif cfg.MULTI_POLICY_MODE == 1: # its own data
                self.multi_policy_masks = torch.zeros(cfg.BATCH_SIZE, cfg.NUM_AGENTS, cfg.NUM_POLICY)
                for i in range(cfg.NUM_AGENTS):
                    for j in range(cfg.NUM_POLICY):
                        if i % cfg.NUM_POLICY == j:
                            self.multi_policy_masks[:, i, j] = 1
            elif cfg.MULTI_POLICY_MODE == 2: # neighbor data
                self.multi_policy_masks = torch.zeros(cfg.BATCH_SIZE, cfg.NUM_AGENTS, cfg.NUM_POLICY)
                for i in range(cfg.NUM_AGENTS):
                    for j in range(cfg.NUM_POLICY):
                        if abs((i % cfg.NUM_POLICY) - j) <= 1:
                            self.multi_policy_masks[:, i, j] = 1
            elif cfg.MULTI_POLICY_MODE == 4:
                assert(cfg.TASK == "smart_streaming")
                self.multi_policy_masks = torch.zeros(cfg.BATCH_SIZE, cfg.NUM_AGENTS, cfg.NUM_POLICY)
                # policy 0~2
                for i in range(3):
                    for j in range(2):
                        for k in range(3):
                            self.multi_policy_masks[:, 12 * j + k, i] = 1
                # policy 3~5
                for i in range(3):
                    for j in range(2):
                        for k in range(3):
                            self.multi_policy_masks[:, 12 * j + k + 3, i + 3] = 1
                # policy 6~8
                for i in range(3):
                    for j in range(2):
                        for k in range(3):
                            self.multi_policy_masks[:, 12 * j + k + 6, i + 6] = 1
                # policy 9~11
                self.multi_policy_masks[:, 0, 9] = 1
                self.multi_policy_masks[:, 12, 9] = 1
                self.multi_policy_masks[:, 9, 9] = 1
                self.multi_policy_masks[:, 21, 9] = 1

                self.multi_policy_masks[:, 3, 10] = 1
                self.multi_policy_masks[:, 15, 10] = 1
                self.multi_policy_masks[:, 10, 10] = 1
                self.multi_policy_masks[:, 22, 10] = 1

                self.multi_policy_masks[:, 6, 11] = 1
                self.multi_policy_masks[:, 18, 11] = 1
                self.multi_policy_masks[:, 11, 11] = 1
                self.multi_policy_masks[:, 23, 11] = 1

            else:
                raise NotImplementedError

            if cfg.CUDA and cfg.EMODE == "pretrain":
                self.multi_policy_masks = self.multi_policy_masks.to(self.device)
            self.multi_policy_masks_length = self.multi_policy_masks.sum()
            if cfg.TASK == "deep_sea_treasure":
                self.multi_policy_masks = self.multi_policy_masks.reshape([-1, cfg.NUM_POLICY, 1])
        
    def update_weights(self):
        print("update weights for SAC")
        self.weights = torch.from_numpy(cfg.MO_WEIGHTS).float()
        print("weights: ", self.weights)
        if cfg.CUDA and cfg.EMODE == "pretrain":
            self.weights = self.weights.to(self.device)

    def random_init(self):
        self.actor.apply(weights_init)
        self.critic_1.apply(weights_init)
        self.critic_2.apply(weights_init)
        self.target_critic_1.apply(weights_init)
        self.target_critic_2.apply(weights_init)

    def act(self, state, rnn_hxs_1=None, rnn_hxs_2=None, masks=None):
        if cfg.IS_RECURRENT:
            if cfg.TASK == "smart_streaming":
                actor_features, rnn_hxs_1, rnn_hxs_2 = self.actor(state, rnn_hxs_1, rnn_hxs_2, masks)
                if cfg.EMODE == "pretrain":
                    dist = self.dist(actor_features)
                    action = dist.sample()
                else:
                    action = actor_features.argmax(2)
                default_action = (cfg.DEFAULT_QUALITY * (state[:, 7, -1] == 0)).float()
                default_action = default_action.reshape(-1, 1).repeat(1, cfg.NUM_POLICY)
                action_condition = (state[:, 7, -1] > 0).float()
                action_condition = action_condition.reshape(-1, 1).repeat(1, cfg.NUM_POLICY)
                action = default_action + action.float() * action_condition
                action = action.long()
                return action, rnn_hxs_1, rnn_hxs_2
        else:
            actor_features = self.actor(state)
            if cfg.EMODE == "pretrain" or cfg.TEST_MODE > 0:
                dist = self.dist(actor_features)
                action = dist.sample()
            else:
                action = actor_features.argmax(2)
            return action

    def cri_1(self, state, rnn_hxs_1, rnn_hxs_2, masks):
        if cfg.TASK == "smart_streaming":
            _, rnn_hxs_1, rnn_hxs_2 = self.critic_1(state, rnn_hxs_1, rnn_hxs_2, masks)
            return rnn_hxs_1, rnn_hxs_2
        else:
            raise NotImplementedError
        
    def cri_2(self, state, rnn_hxs_1, rnn_hxs_2, masks):
        if cfg.TASK == "smart_streaming":
            _, rnn_hxs_1, rnn_hxs_2 = self.critic_2(state, rnn_hxs_1, rnn_hxs_2, masks)
            return rnn_hxs_1, rnn_hxs_2
        else:
            raise NotImplementedError

    def train(self, rollouts):
        if cfg.IS_RECURRENT:
            # states [cfg.BATCH_SIZE + 1, cfg.NUM_AGENTS, state_shape]
            states = rollouts[0]
            # actions [cfg.BATCH_SIZE, cfg.NUM_AGENTS, 1]
            actions = rollouts[1]
            # actions_ [cfg.BATCH_SIZE, cfg.NUM_AGENTS, cfg.NUM_POLICY, 1]
            actions_ = actions.repeat(1, 1, cfg.NUM_POLICY).reshape([cfg.BATCH_SIZE, cfg.NUM_AGENTS, cfg.NUM_POLICY, 1]).long()
            # rewards [cfg.BATCH_SIZE, cfg.NUM_AGENTS, cfg.NUM_REWARD]
            rewards = rollouts[2]
            rewards = torch.matmul(rewards, self.weights)
            # rewards [cfg.BATCH_SIZE, cfg.NUM_AGENTS, cfg.NUM_POLICY, 1]
            rewards = rewards.reshape([cfg.BATCH_SIZE, cfg.NUM_AGENTS, cfg.NUM_POLICY])
            # dones [cfg.BATCH_SIZE, cfg.NUM_AGENTS, 1]
            dones = rollouts[3]
            # dones_ [cfg.BATCH_SIZE, cfg.NUM_AGENTS, cfg.NUM_POLICY]
            dones_ = dones.repeat(1, 1, cfg.NUM_POLICY)
            # masks [cfg.BATCH_SIZE + 1, cfg.NUM_AGENTS, 1]
            masks = rollouts[4]
            # next_states [cfg.BATCH_SIZE, cfg.NUM_AGENTS, state_shape]
            # next_states = rollouts[2]
            
            with torch.no_grad():
                next_actor_probs = self.actor.sequentially_forward(states, masks)
                # next_actor_probs [cfg.BATCH_SIZE, cfg.NUM_AGENTS, cfg.NUM_POLICY, cfg.A_DIM]
                next_actor_probs = next_actor_probs[1:]
                next_dist = self.dist(next_actor_probs)
                # next_actions [cfg.BATCH_SIZE, cfg.NUM_AGENTS, cfg.NUM_POLICY]
                next_actions = next_dist.sample()
                # next_actions [cfg.BATCH_SIZE, cfg.NUM_AGENTS, cfg.NUM_POLICY, 1]
                next_actions_ = next_actions.reshape([cfg.BATCH_SIZE, cfg.NUM_AGENTS, cfg.NUM_POLICY, 1])
                # log_pi_q [cfg.BATCH_SIZE, cfg.NUM_AGENTS, cfg.NUM_POLICY]
                log_pi_q = next_dist.log_prob(next_actions)
                
                # target_q1_pred [cfg.BATCH_SIZE + 1, cfg.NUM_AGENTS, cfg.NUM_POLICY, cfg.A_DIM]
                target_q1_pred = self.target_critic_1.sequentially_forward(states, masks)
                # target_q1_pred [cfg.BATCH_SIZE, cfg.NUM_AGENTS, cfg.NUM_POLICY, cfg.A_DIM]
                target_q1_pred = target_q1_pred[1:]
                # target_q1_pred [cfg.BATCH_SIZE, cfg.NUM_AGENTS, cfg.NUM_POLICY]
                target_q1_pred = target_q1_pred.gather(3, next_actions_).reshape(cfg.BATCH_SIZE, cfg.NUM_AGENTS, cfg.NUM_POLICY)
                target_q2_pred = self.target_critic_2.sequentially_forward(states, masks)
                target_q2_pred = target_q2_pred[1:]
                target_q2_pred = target_q2_pred.gather(3, next_actions_).reshape(cfg.BATCH_SIZE, cfg.NUM_AGENTS, cfg.NUM_POLICY)
                
                target_q_values = torch.min(target_q1_pred, target_q2_pred) - cfg.ENTROPY_WEIGHT * log_pi_q
                # q_target [cfg.BATCH_SIZE, cfg.NUM_AGENTS, cfg.NUM_POLICY]
                q_target = rewards + cfg.GAMMA * ((1. - dones_) * target_q_values)

            ##########
            # Q_pred #
            ##########
            q1_pred = self.critic_1.sequentially_forward(states, masks)
            q2_pred = self.critic_2.sequentially_forward(states, masks)
            # new_q_pred is predicted for Policy Loss
            new_q_pred = torch.min(q1_pred, q2_pred).detach()
            # q1_pred [cfg.BATCH_SIZE, cfg.NUM_AGENTS, cfg.NUM_POLICY, cfg.A_DIM]
            q1_pred = q1_pred[:-1]
            # q1_pred_ [cfg.BATCH_SIZE, cfg.NUM_AGENTS, cfg.NUM_POLICY]
            q1_pred_ = q1_pred.gather(3, actions_).reshape(cfg.BATCH_SIZE, cfg.NUM_AGENTS, cfg.NUM_POLICY)

            q2_pred = q2_pred[:-1]
            q2_pred_ = q2_pred.gather(3, actions_).reshape(cfg.BATCH_SIZE, cfg.NUM_AGENTS, cfg.NUM_POLICY)

            ##########
            # Q Loss #
            ##########
            if cfg.NUM_POLICY == 1:
                qf1_loss = F.mse_loss(q1_pred_, q_target)
                qf2_loss = F.mse_loss(q2_pred_, q_target)
            else:
                qf1_loss = (((q1_pred_ - q_target) ** 2) * self.multi_policy_masks).sum() / self.multi_policy_masks_length
                qf2_loss = (((q2_pred_ - q_target) ** 2) * self.multi_policy_masks).sum() / self.multi_policy_masks_length

            ###############
            # Policy Loss #
            ###############
            # actor_probs [cfg.BATCH_SIZE, cfg.NUM_AGENTS, cfg.NUM_POLICY, cfg.ACTION_DIM]
            actor_probs = self.actor.sequentially_forward(states, masks)
            dist = self.dist(actor_probs)
            policy_loss = 0

            for i in range(cfg.A_DIM):
                # actions_i: [cfg.BATCH_SIZE, cfg.NUM_AGENTS, cfg.NUM_POLICY]
                actions_i = (torch.zeros(cfg.BATCH_SIZE + 1, cfg.NUM_AGENTS, cfg.NUM_POLICY) + i).long()
                if cfg.CUDA:
                    actions_i = actions_i.to(self.device)
                # log_pi_i: [cfg.BATCH_SIZE, cfg.NUM_AGENTS, cfg.NUM_POLICY]
                log_pi_i = dist.log_prob(actions_i)
                # new_q_pred_i: [cfg.BATCH_SIZE, cfg.NUM_AGENTS, cfg.NUM_POLICY]
                new_q_pred_i = new_q_pred[:, :, :, i]
                # weight_i: [cfg.BATCH_SIZE, cfg.NUM_AGENTS, cfg.NUM_POLICY]
                weight_i = cfg.ENTROPY_WEIGHT * log_pi_i - new_q_pred_i + cfg.ENTROPY_WEIGHT
                policy_loss += (weight_i.detach() * log_pi_i.exp()).mean()

            policy_loss = policy_loss / cfg.A_DIM
            
            ###################
            # Update networks #
            ###################
            self.critic_opt_1.zero_grad()
            qf1_loss.backward()
            self.critic_opt_1.step()

            self.critic_opt_2.zero_grad()
            qf2_loss.backward()
            self.critic_opt_2.step()
            
            self.actor_opt.zero_grad()
            policy_loss.backward()
            self.actor_opt.step()
            alpha_loss = torch.tensor(0.)

            # Soft Updates
            soft_update_from_to(
                self.critic_1, self.target_critic_1, cfg.SOFT_TARGET_TAU
            )
            soft_update_from_to(
                self.critic_2, self.target_critic_2, cfg.SOFT_TARGET_TAU
            )
            return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item()

        else:
            # states [cfg.BATCH_SIZE, cfg.NUM_AGENTS, state_shape]
            states = rollouts[0]
            # states_ [cfg.BATCH_SIZE * cfg.NUM_AGENTS, state_shape]
            states_ = states.reshape(cfg.BATCH_SIZE * cfg.NUM_AGENTS, -1)
            
            # actions [cfg.BATCH_SIZE, cfg.NUM_AGENTS, 1]
            actions = rollouts[1]
            # actions_ [cfg.BATCH_SIZE * cfg.NUM_AGENTS, cfg.NUM_POLICY, 1]
            actions_ = actions.reshape(cfg.BATCH_SIZE * cfg.NUM_AGENTS, 1, 1).repeat(1, cfg.NUM_POLICY, 1)
            
            # next_states [cfg.BATCH_SIZE, cfg.NUM_AGENTS, state_shape]
            next_states = rollouts[2]
            # next_states_ [cfg.BATCH_SIZE * cfg.NUM_AGENTS, state_shape]
            next_states_ = next_states.reshape(cfg.BATCH_SIZE * cfg.NUM_AGENTS, -1)
            
            # rewards [cfg.BATCH_SIZE, cfg.NUM_AGENTS, cfg.NUM_REWARD]
            rewards = rollouts[3]
            if cfg.SCALARIZATION_METHOD == 1:
                rewards = torch.matmul(rewards, self.weights)
                # rewards [cfg.BATCH_SIZE * cfg.NUM_AGENTS, cfg.NUM_POLICY, 1]
                rewards = rewards.reshape([cfg.BATCH_SIZE * cfg.NUM_AGENTS, cfg.NUM_POLICY, 1])
            elif cfg.SCALARIZATION_METHOD == 2:
                rewards = rewards.reshape(cfg.BATCH_SIZE, cfg.NUM_AGENTS, 1, cfg.NUM_REWARD).repeat(1, 1, cfg.NUM_POLICY, 1)
                ths = self.ths.reshape(1, 1, cfg.NUM_POLICY, cfg.NUM_REWARD).repeat(cfg.BATCH_SIZE, cfg.NUM_AGENTS, 1, 1)
                # rewards [cfg.BATCH_SIZE, cfg.NUM_AGENTS, cfg.NUM_POLICY, cfg.NUM_REWARD]
                rewards[:, :, :, 1] = torch.min(rewards[:, :, :, 1], ths[:, :, :, 1])
                rewards[:, :, :, 0] = rewards[:, :, :, 0] * 0.01
                rewards = rewards.sum(-1)
                rewards = rewards.reshape([cfg.BATCH_SIZE * cfg.NUM_AGENTS, cfg.NUM_POLICY, 1])
            else:
                raise NotImplementedError
            
            # dones [cfg.BATCH_SIZE, cfg.NUM_AGENTS, 1]
            dones = rollouts[4]
            # dones_ [cfg.BATCH_SIZE * cfg.NUM_AGENTS, cfg.NUM_POLICY, 1]
            dones_ = dones.reshape(cfg.BATCH_SIZE * cfg.NUM_AGENTS, 1, 1).repeat(1, cfg.NUM_POLICY, 1)
            
            with torch.no_grad():
                ######################
                # Q Target, log_pi_q #
                ######################
                # next_actor_probs [cfg.BATCH_SIZE * cfg.NUM_AGENTS, cfg.NUM_POLICY, cfg.ACTION_DIM]
                next_actor_probs = self.actor(next_states_)
                next_dist = self.dist(next_actor_probs)
                next_actions = next_dist.sample()
                # next_actions_ [cfg.BATCH_SIZE * cfg.NUM_AGENTS, cfg.NUM_POLICY, 1]
                next_actions_ = next_actions.reshape(cfg.BATCH_SIZE * cfg.NUM_AGENTS, cfg.NUM_POLICY, 1)
                # log_pi_q [cfg.BATCH_SIZE * cfg.NUM_AGENTS, cfg.NUM_POLICY, 1]
                log_pi_q = next_dist.log_prob(next_actions).reshape(cfg.BATCH_SIZE * cfg.NUM_AGENTS, cfg.NUM_POLICY, 1)
                
                # target_q1_pred [cfg.BATCH_SIZE * cfg.NUM_AGENTS, cfg.NUM_POLICY, cfg.A_DIM]
                target_q1_pred = self.target_critic_1(next_states_)
                # target_q1_pred_ [cfg.BATCH_SIZE * cfg.NUM_AGENTS, cfg.NUM_POLICY, 1]
                target_q1_pred_ = target_q1_pred.gather(2, next_actions_)
                target_q2_pred = self.target_critic_2(next_states_)
                target_q2_pred_ = target_q2_pred.gather(2, next_actions_)
                # self.alpha [cfg.NUM_POLICY, 1]
                # target_q_values [cfg.BATCH_SIZE * cfg.NUM_AGENTS, cfg.NUM_POLICY, 1]
                target_q_values = torch.min(target_q1_pred_, target_q2_pred_) - self.alpha * log_pi_q
                
                # q_target [cfg.BATCH_SIZE * cfg.NUM_AGENTS, cfg.NUM_POLICY, 1]
                q_target = rewards + cfg.GAMMA * ((1. - dones_) * target_q_values)
                if cfg.AUTOMATIC_ENTROPY_TUNING:
                    # alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
                    # alpha_loss_weights [cfg.BATCH_SIZE * cfg.NUM_AGENTS, cfg.NUM_POLICY]
                    alpha_loss_weights = (log_pi_q.reshape(cfg.BATCH_SIZE * cfg.NUM_AGENTS, cfg.NUM_POLICY) + next_dist.entropy()).detach()
                
            ##########
            # Q_pred #
            ##########
            # q1_pred [cfg.BATCH_SIZE * cfg.NUM_AGENTS, cfg.NUM_POLICY, cfg.ACTION_DIM]
            q1_pred = self.critic_1(states_)
            # q1_pred_ [cfg.BATCH_SIZE * cfg.NUM_AGENTS, cfg.NUM_POLICY, 1]
            q1_pred_ = q1_pred.gather(2, actions_)
            q2_pred = self.critic_2(states_)
            q2_pred_ = q2_pred.gather(2, actions_)

            ##########
            # Q Loss #
            ##########
            if cfg.NUM_POLICY == 1:
                qf1_loss = F.mse_loss(q1_pred_, q_target)
                qf2_loss = F.mse_loss(q2_pred_, q_target)
            else:
                qf1_loss = (((q1_pred_ - q_target) ** 2) * self.multi_policy_masks).sum() / self.multi_policy_masks_length
                qf2_loss = (((q2_pred_ - q_target) ** 2) * self.multi_policy_masks).sum() / self.multi_policy_masks_length
            
            ###############
            # Policy Loss #
            ###############
            new_q1_pred = self.critic_1(states_)
            new_q2_pred = self.critic_2(states_)
            # new_q_pred [cfg.BATCH_SIZE * cfg.NUM_AGENTS, cfg.NUM_POLICY, cfg.A_DIM]
            new_q_pred = torch.min(new_q1_pred, new_q2_pred)

            # actor_probs [cfg.BATCH_SIZE * cfg.NUM_AGENTS, cfg.NUM_POLICY, cfg.ACTION_DIM]
            actor_probs = self.actor(states_)
            dist = self.dist(actor_probs)
            policy_loss = 0

            for i in range(cfg.A_DIM):
                # actions_i: [cfg.BATCH_SIZE * cfg.NUM_AGENTS, cfg.NUM_POLICY]
                actions_i = (torch.zeros(cfg.BATCH_SIZE * cfg.NUM_AGENTS, cfg.NUM_POLICY) + i).long()
                # log_pi_i: [cfg.BATCH_SIZE * cfg.NUM_AGENTS, cfg.NUM_POLICY]
                log_pi_i = dist.log_prob(actions_i)
                # new_q_pred_i: [cfg.BATCH_SIZE * cfg.NUM_AGENTS, cfg.NUM_POLICY]
                new_q_pred_i = new_q_pred[:, :, i]
                # weight_i: [cfg.BATCH_SIZE * cfg.NUM_AGENTS, cfg.NUM_POLICY]
                weight_i = self.alpha.squeeze() * log_pi_i - new_q_pred_i + self.alpha.squeeze()
                policy_loss += (weight_i.detach() * log_pi_i.exp()).mean()
                
            policy_loss = policy_loss / cfg.A_DIM
                
            ###################
            # Update networks #
            ###################
            self.critic_opt_1.zero_grad()
            qf1_loss.backward()
            self.critic_opt_1.step()

            self.critic_opt_2.zero_grad()
            qf2_loss.backward()
            self.critic_opt_2.step()
            
            self.actor_opt.zero_grad()
            policy_loss.backward()
            self.actor_opt.step()
            
            if cfg.AUTOMATIC_ENTROPY_TUNING:
                # alpha_loss = -(self.log_alpha * alpha_loss_weights).mean()
                alpha_loss = -(self.alpha.squeeze() * alpha_loss_weights).mean()

                self.alpha_optim.zero_grad()
                alpha_loss.backward()
                self.alpha_optim.step()

                # self.alpha = self.log_alpha.exp()
            else:
                alpha_loss = torch.tensor(0.)

            # Soft Updates
            soft_update_from_to(
                self.critic_1, self.target_critic_1, cfg.SOFT_TARGET_TAU
            )
            soft_update_from_to(
                self.critic_2, self.target_critic_2, cfg.SOFT_TARGET_TAU
            )
            return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), self.alpha.mean()
        
    def load_model(self, model_name="default"):
        if model_name == "default":
            if cfg.IS_REPEAT == 0:
                actor_model = osp.abspath(osp.join(cfg.SAVE_DIR, "actor_ep_" + str(cfg.MODEL_EPOCHES)))
                self.actor.load_state_dict(torch.load(actor_model))
                critic_1_model = osp.abspath(osp.join(cfg.SAVE_DIR, "critic_1_ep_" + str(cfg.MODEL_EPOCHES)))
                self.critic_1.load_state_dict(torch.load(critic_1_model))
                critic_2_model = osp.abspath(osp.join(cfg.SAVE_DIR, "critic_2_ep_" + str(cfg.MODEL_EPOCHES)))
                self.critic_2.load_state_dict(torch.load(critic_2_model))
            else:
                # 000_critic_2_ep_400
                actor_model = osp.abspath(osp.join(cfg.SAVE_DIR, "%03d_actor_ep_%d"%(cfg.MODEL_REPEAT_IDX, cfg.MODEL_EPOCHES)))
                self.actor.load_state_dict(torch.load(actor_model))
                critic_1_model = osp.abspath(osp.join(cfg.SAVE_DIR, "%03d_critic_1_ep_%d"%(cfg.MODEL_REPEAT_IDX, cfg.MODEL_EPOCHES)))
                self.critic_1.load_state_dict(torch.load(critic_1_model))
                critic_2_model = osp.abspath(osp.join(cfg.SAVE_DIR, "%03d_critic_2_ep_%d"%(cfg.MODEL_REPEAT_IDX, cfg.MODEL_EPOCHES)))
                self.critic_2.load_state_dict(torch.load(critic_2_model))
        else:
            actor_model = osp.abspath(osp.join(cfg.SAVE_DIR, model_name))
            self.actor.load_state_dict(torch.load(actor_model))
            
class SoftActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size, num_policy):
        super(SoftActor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.num_policy = num_policy
        if cfg.ONE_LAYER_NET:
            self.fcl0 = nn.Linear(state_dim[0], num_policy * action_dim)
        else:
            self.fcl0 = nn.Linear(state_dim[0], self.hidden_size)
            self.fcl1 = nn.Linear(hidden_size, num_policy * action_dim) # multi-policy
        if cfg.ACTIVATION_FUNCTION == "relu":
            self.act = F.relu
        elif cfg.ACTIVATION_FUNCTION == "tanh":
            self.act = F.tanh
        else:
            self.act = F.relu
        self.distribution = torch.distributions.Categorical
    def forward(self, state):
        if cfg.ONE_LAYER_NET:
            x = self.fcl0(state)
        else:
            x = F.relu(self.fcl0(state))
            x = self.fcl1(x)
        x_shape = x.shape # N * (cfg.NUM_POLICY * action_dim)
        x = x.reshape([x_shape[0], self.num_policy, self.action_dim]) # N * cfg.NUM_POLICY * action_dim
        probs = F.softmax(x, dim=2) # N * cfg.NUM_POLICY * action_dim
        return probs

class SoftCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size):
        super(SoftCritic, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        if cfg.ONE_LAYER_NET:
            self.fcl0 = nn.Linear(state_dim[0], cfg.NUM_POLICY * action_dim)
        else:
            self.fcl0 = nn.Linear(state_dim[0], hidden_size)
            self.fcl1 = nn.Linear(hidden_size, cfg.NUM_POLICY * self.action_dim)
        if cfg.ACTIVATION_FUNCTION == "relu":
            self.act = F.relu
        elif cfg.ACTIVATION_FUNCTION == "tanh":
            self.act = F.tanh
        else:
            self.act = F.relu
        
    def forward(self, state):
        if cfg.ONE_LAYER_NET:
            val = self.fcl0(state)
        else:
            x = F.relu(self.fcl0(state))
            val = self.fcl1(x)
        val_shape = val.shape
        val = val.reshape([val_shape[0], cfg.NUM_POLICY, self.action_dim])
        return val

class RecSoftActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size_1, hidden_size_2, num_policy):
        super(RecSoftActor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.num_policy = num_policy

        input_channel = 32
        assert(not cfg.INPUT_WITHOUT_CHUNK_SIZE or not cfg.INPUT_WITHOUT_QOE)
        if cfg.INPUT_WITHOUT_CHUNK_SIZE or cfg.INPUT_WITHOUT_QOE:
            input_channel -= (cfg.A_DIM + 1)
        hidden_size_0 = input_channel
        
        self.gru0 = nn.GRUCell(hidden_size_0, self.hidden_size_1)
        self.gru1 = nn.GRUCell(self.hidden_size_1, self.hidden_size_2)
        self.linear3 = nn.Linear(self.hidden_size_2, self.hidden_size_2)
        self.fcl_final = nn.Linear(self.hidden_size_2, self.action_dim * self.num_policy)
        
        for name, param in self.gru0.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param)
        for name, param in self.gru1.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param)
        if cfg.CUDA and cfg.TASK == "pretrain":
            self.device = torch.device(cfg.CUDA_DEVICE)
        
    def forward(self, state, rnn_hxs_1, rnn_hxs_2, masks):
        #  1st row: chunk size (kB) of pask k chunks
        #  2nd row: perceptual quality of pask k chunks
        #  3rd row: buffer size (s) of pask k chunks
        #  4th row: bandwidth (kB/ms) of past k chunks
        #  5th row: download time (s) of past k chunks
        #  6th row: bitrate array (MB) of next chunk (for num_rep representations)
        #  7th row: perceptual quality array of next chunk (for num_rep representations)
        #  8th row: number of downloaded chunks
        #  9th row: total number of chunks
        # 10th row: bitrate level of past k chunks
        MB_to_kbps = cfg.B_IN_KB * cfg.BITS_IN_BYTE / cfg.VIDEO_CHUNK_LEN_S
        kB_to_kbps = cfg.BITS_IN_BYTE / cfg.VIDEO_CHUNK_LEN_S
        # last chunk size
        # perceptual quality
        # buffer level
        # last bandwidth metric
        # last download time
        # available bitrate array
        # available perceptual score
        # ratio of remaining chunks
        if cfg.INPUT_WITHOUT_CHUNK_SIZE:
            x = torch.cat((state[:, 1:2, -1] / cfg.VMAF_NORM_FACTOR, \
                   state[:, 2:3, -1] / cfg.BUFFER_NORM_FACTOR, \
                   state[:, 3:4, -1], \
                   state[:, 4:5, -1], \
                   state[:, 6:7, :cfg.A_DIM].squeeze(1) / cfg.VMAF_NORM_FACTOR, \
                   1 - state[:, 7:8, -1] / (state[:, 8:9, -1] + 1e-6)), 1)
        elif cfg.INPUT_WITHOUT_QOE:
            x = torch.cat((state[:, 0:1, -1] * kB_to_kbps / float(np.max(cfg.VIDEO_BIT_RATE)) * cfg.BITRATE_NORM_FACTOR, \
                   state[:, 2:3, -1] / cfg.BUFFER_NORM_FACTOR, \
                   state[:, 3:4, -1], \
                   state[:, 4:5, -1], \
                   state[:, 5:6, :cfg.A_DIM].squeeze(1) * MB_to_kbps / float(np.max(cfg.VIDEO_BIT_RATE)) * cfg.BITRATE_NORM_FACTOR, \
                   1 - state[:, 7:8, -1] / (state[:, 8:9, -1] + 1e-6)), 1)
        else:    
            x = torch.cat((state[:, 0:1, -1] * kB_to_kbps / float(np.max(cfg.VIDEO_BIT_RATE)) * cfg.BITRATE_NORM_FACTOR, \
                   state[:, 1:2, -1] / cfg.VMAF_NORM_FACTOR, \
                   state[:, 2:3, -1] / cfg.BUFFER_NORM_FACTOR, \
                   state[:, 3:4, -1], \
                   state[:, 4:5, -1], \
                   state[:, 5:6, :cfg.A_DIM].squeeze(1) * MB_to_kbps / float(np.max(cfg.VIDEO_BIT_RATE)) * cfg.BITRATE_NORM_FACTOR, \
                   state[:, 6:7, :cfg.A_DIM].squeeze(1) / cfg.VMAF_NORM_FACTOR, \
                   1 - state[:, 7:8, -1] / (state[:, 8:9, -1] + 1e-6)), 1)
        rnn_hxs_1 = self.gru0(x, rnn_hxs_1 * masks.repeat(1, self.hidden_size_1))
        rnn_hxs_2 = self.gru1(rnn_hxs_1, rnn_hxs_2 * masks.repeat(1, self.hidden_size_2))
        if cfg.ADD_LINEAR_LAYER_3:
            rnn_hxs_2_ = F.relu(self.linear3(rnn_hxs_2))
            probs = self.fcl_final(rnn_hxs_2_)
        else:
            probs = self.fcl_final(rnn_hxs_2)
        probs_shape = probs.shape
        probs = probs.reshape([probs_shape[0], self.num_policy, self.action_dim])
        probs = F.softmax(probs, dim=2) # N * cfg.NUM_POLICY * action_dim
        return probs, rnn_hxs_1, rnn_hxs_2

    def sequentially_forward(self, states, mask):
        hnn_1 = torch.zeros(cfg.NUM_AGENTS, cfg.DIM_FIRST_LAYER)
        hnn_2 = torch.zeros(cfg.NUM_AGENTS, cfg.DIM_SECOND_LAYER)
        hnn_1.fill_(0)
        hnn_2.fill_(0)
        if cfg.CUDA:
            hnn_1 = hnn_1.to(self.device)
            hnn_2 = hnn_2.to(self.device)
        action_probs = []
        for i in range(states.shape[0]):
            action_prob, hnn_1, hnn_2 = self.forward(states[i], hnn_1, hnn_2, mask[i])
            action_probs.append(action_prob)
        return torch.stack(action_probs)

class RecSoftCritic(nn.Module): # not ready
    def __init__(self, state_dim, action_dim, hidden_size_1, hidden_size_2):
        super(RecSoftCritic, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        input_channel = 32
        assert(not cfg.INPUT_WITHOUT_CHUNK_SIZE or not cfg.INPUT_WITHOUT_QOE)
        if cfg.INPUT_WITHOUT_CHUNK_SIZE or cfg.INPUT_WITHOUT_QOE:
            input_channel -= (cfg.A_DIM + 1)
        hidden_size_0 = input_channel
        
        self.gru0 = nn.GRUCell(hidden_size_0, self.hidden_size_1)
        self.gru1 = nn.GRUCell(self.hidden_size_1, self.hidden_size_2)
        self.linear3 = nn.Linear(self.hidden_size_2, self.hidden_size_2)
        self.fcl_final = nn.Linear(self.hidden_size_2, self.action_dim * cfg.NUM_POLICY)
        
        for name, param in self.gru0.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.kaiming_uniform_(param, mode="fan_in", nonlinearity="relu")
        for name, param in self.gru1.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.kaiming_uniform_(param, mode="fan_in", nonlinearity="relu")
        if cfg.CUDA:
            self.device = torch.device(cfg.CUDA_DEVICE)

    def forward(self, state, rnn_hxs_1, rnn_hxs_2, masks):
        #  1st row: chunk size (kB) of pask k chunks
        #  2nd row: perceptual quality of pask k chunks
        #  3rd row: buffer size (s) of pask k chunks
        #  4th row: bandwidth (kB/ms) of past k chunks
        #  5th row: download time (s) of past k chunks
        #  6th row: bitrate array (MB) of next chunk (for num_rep representations)
        #  7th row: perceptual quality array of next chunk (for num_rep representations)
        #  8th row: number of downloaded chunks
        #  9th row: total number of chunks
        # 10th row: bitrate level of past k chunks
        MB_to_kbps = cfg.B_IN_KB * cfg.BITS_IN_BYTE / cfg.VIDEO_CHUNK_LEN_S
        kB_to_kbps = cfg.BITS_IN_BYTE / cfg.VIDEO_CHUNK_LEN_S
        # last chunk size
        # perceptual quality
        # buffer level
        # last bandwidth metric
        # last download time
        # available bitrate array
        # available perceptual score
        # ratio of remaining chunks
        if cfg.INPUT_WITHOUT_CHUNK_SIZE:
            x = torch.cat((state[:, 1:2, -1] / cfg.VMAF_NORM_FACTOR, \
                   state[:, 2:3, -1] / cfg.BUFFER_NORM_FACTOR, \
                   state[:, 3:4, -1], \
                   state[:, 4:5, -1], \
                   state[:, 6:7, :cfg.A_DIM].squeeze(1) / cfg.VMAF_NORM_FACTOR, \
                   1 - state[:, 7:8, -1] / (state[:, 8:9, -1] + 1e-6)), 1)
        elif cfg.INPUT_WITHOUT_QOE:
            x = torch.cat((state[:, 0:1, -1] * kB_to_kbps / float(np.max(cfg.VIDEO_BIT_RATE)) * cfg.BITRATE_NORM_FACTOR, \
                   state[:, 2:3, -1] / cfg.BUFFER_NORM_FACTOR, \
                   state[:, 3:4, -1], \
                   state[:, 4:5, -1], \
                   state[:, 5:6, :cfg.A_DIM].squeeze(1) * MB_to_kbps / float(np.max(cfg.VIDEO_BIT_RATE)) * cfg.BITRATE_NORM_FACTOR, \
                   1 - state[:, 7:8, -1] / (state[:, 8:9, -1] + 1e-6)), 1)
        else:
            x = torch.cat((state[:, 0:1, -1] * kB_to_kbps / float(np.max(cfg.VIDEO_BIT_RATE)) * cfg.BITRATE_NORM_FACTOR, \
                   state[:, 1:2, -1] / cfg.VMAF_NORM_FACTOR, \
                   state[:, 2:3, -1] / cfg.BUFFER_NORM_FACTOR, \
                   state[:, 3:4, -1], \
                   state[:, 4:5, -1], \
                   state[:, 5:6, :cfg.A_DIM].squeeze(1) * MB_to_kbps / float(np.max(cfg.VIDEO_BIT_RATE)) * cfg.BITRATE_NORM_FACTOR, \
                   state[:, 6:7, :cfg.A_DIM].squeeze(1) / cfg.VMAF_NORM_FACTOR, \
                   1 - state[:, 7:8, -1] / (state[:, 8:9, -1] + 1e-6)), 1)
        rnn_hxs_1 = self.gru0(x, rnn_hxs_1 * masks.repeat(1, self.hidden_size_1))
        rnn_hxs_2 = self.gru1(rnn_hxs_1, rnn_hxs_2 * masks.repeat(1, self.hidden_size_2))
        if cfg.ADD_LINEAR_LAYER_3:
            rnn_hxs_2_ = F.relu(self.linear3(rnn_hxs_2))
            val = self.fcl_final(rnn_hxs_2_)
        else:
            val = self.fcl_final(rnn_hxs_2)
        val_shape = val.shape
        val = val.reshape([val_shape[0], cfg.NUM_POLICY, self.action_dim])
        return val, rnn_hxs_1, rnn_hxs_2
    
    def sequentially_forward(self, states, mask):
        hnn_1 = torch.zeros(cfg.NUM_AGENTS, cfg.DIM_FIRST_LAYER)
        hnn_2 = torch.zeros(cfg.NUM_AGENTS, cfg.DIM_SECOND_LAYER)
        hnn_1.fill_(0)
        hnn_2.fill_(0)
        if cfg.CUDA:
            hnn_1 = hnn_1.to(self.device)
            hnn_2 = hnn_2.to(self.device)
        vals = []
        for i in range(states.shape[0]):
            val, hnn_1, hnn_2 = self.forward(states[i], hnn_1, hnn_2, mask[i])
            vals.append(val)
        return torch.stack(vals)
    
class RecSoftActor2(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size, num_policy):
        super(RecSoftActor2, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.num_policy = num_policy

        self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.gru0 = nn.GRUCell(32 * 4 * 4, self.hidden_size)
        self.fcl_final = nn.Linear(self.hidden_size, self.action_dim * self.num_policy)
        
        for name, param in self.gru0.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param)
        if cfg.CUDA:
            self.device = torch.device(cfg.CUDA_DEVICE)

    def forward(self, state, rnn_hxs, masks):
        # state: [NUM_AGENTS, 3, 60, 64]
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        # x: [NUM_AGENTS, 32, 4, 4]
        rnn_hxs = self.gru0(x.view(x.size(0), -1), rnn_hxs * masks.repeat(1, self.hidden_size))
        probs = self.fcl_final(rnn_hxs)
        probs_shape = probs.shape
        probs = probs.reshape([probs_shape[0], self.num_policy, self.action_dim])
        probs = F.softmax(probs, dim=2) # N * cfg.NUM_POLICY * action_dim
        return probs, rnn_hxs

    def sequentially_forward(self, states, mask):
        hnn = torch.zeros(cfg.NUM_AGENTS, cfg.DIM_FIRST_LAYER)
        hnn.fill_(0)
        if cfg.CUDA:
            hnn = hnn.to(self.device)
        action_probs = []
        for i in range(states.shape[0]):
            action_prob, hnn = self.forward(states[i], hnn, mask[i])
            action_probs.append(action_prob)
        return torch.stack(action_probs)

class RecSoftCritic2(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size, num_policy):
        super(RecSoftCritic2, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.num_policy = num_policy

        self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.gru0 = nn.GRUCell(32 * 4 * 4, self.hidden_size)
        self.fcl_final = nn.Linear(self.hidden_size, self.action_dim * self.num_policy)
        
        for name, param in self.gru0.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param)
        if cfg.CUDA:
            self.device = torch.device(cfg.CUDA_DEVICE)

    def forward(self, state, rnn_hxs, masks):
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        rnn_hxs = self.gru0(x.view(x.size(0), -1), rnn_hxs * masks.repeat(1, self.hidden_size))
        val = self.fcl_final(rnn_hxs)
        val_shape = val.shape
        val = val.reshape([val_shape[0], cfg.NUM_POLICY, self.action_dim])
        return val, rnn_hxs
    
    def sequentially_forward(self, states, mask):
        hnn = torch.zeros(cfg.NUM_AGENTS, cfg.DIM_FIRST_LAYER)
        hnn.fill_(0)
        if cfg.CUDA:
            hnn = hnn.to(self.device)
        vals = []
        for i in range(states.shape[0]):
            val, hnn = self.forward(states[i], hnn, mask[i])
            vals.append(val)
        return torch.stack(vals)