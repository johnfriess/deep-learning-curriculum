import torch
import hydra
import gymnasium as gym
import numpy as np
from gymnasium.spaces.utils import flatdim
from jaxtyping import Int, Float

class PPO(torch.nn.Module):
    def __init__(self, obs_dim=4, hidden_dim=64, action_dim=2, epsilon=0.2):
        self.epsilon = epsilon
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU()
        )
        self.policy = torch.nn.Linear(hidden_dim, action_dim)
        self.value = torch.nn.Linear(hidden_dim, 1)

    def forward(self, state: Float[torch.Tensor, "batch obs_dim1 obs_dim2 ..."]) -> Float[torch.Tensor, "batch hidden_dim"]:
        B = state.shape[0]
        state = state.view(B, -1)
        return self.encoder(state)

    def get_policy(self, state: Float[torch.Tensor, "batch obs_dim1 obs_dim2 ..."]) -> Float[torch.Tensor, "batch action_dim"]:
        state = self.forward(state)
        return self.policy(state)
    
    def get_prob(self, state: Float[torch.Tensor, "batch obs_dim1 obs_dim2 ..."], action: Float[torch.Tensor, "batch action"]) -> Float[torch.Tensor, "batch prob"]:
        logits = self.get_policy(state)
        probs = torch.softmax(logits, dim=1)
        return probs[action]

    def get_action(self, state: Float[torch.Tensor, "batch obs_dim1 obs_dim2 ..."]) -> Float[torch.Tensor, "batch action"]:
        logits = self.get_policy(state)
        probs = torch.softmax(logits, dim=1)
        return torch.multinomial(probs, num_samples=1)
    
    def get_value(self, state: Float[torch.Tensor, "batch obs_dim1 obs_dim2 ..."]) -> Float[torch.Tensor, "batch value"]:
        state = self.forward(state)
        return self.value(state)

    def compute_loss(self, ratios: Float[torch.Tensor, "t"], advantages: Float[torch.Tensor, "t"]) -> Float[torch.Tensor, "t"]:
        return torch.mean(torch.min(ratios * advantages, torch.clip(ratios, 1 - self.epsilon, 1 + self.epsilon)))

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(config):
    env = gym.make(config.env, render_mode="human")
    obs_dim = flatdim(env.observation_space)
    action_dim = flatdim(env.action_space)

    obs, info = env.reset(seed=0)

    model = PPO(
        obs_dim=obs_dim,
        hidden_dim=64,
        action_dim=action_dim,
        epsilon=config.epsilon
    )
    optimizer = torch.optim.Adam(model.parameters(), config.lr)

    total_timesteps = 0
    while total_timesteps < config.total_timesteps:
        # collect TIMESTEPS_PER_ROLLOUT
        deltas = []
        advantages = []
        ratios = []
        old_probs = []
        for rollout_step in range(config.timesteps_per_rollout):
            action = model.get_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                deltas.append(reward - model.get_value(obs))
                obs, info = env.reset()
            else:
                deltas.append(reward + config.gamma * model.get_value(next_obs) - model.get_value(obs))
                obs = next_obs
            
            total_timesteps += 1

        advantages.append(0)
        for delta in deltas[::-1]:
            advantages.append(delta + config.gamma * config.lam * advantages[-1])
        advantages.pop(0)
        advantages = advantages[::-1]
        
        # update policy
        for epoch in config.epochs_per_rollout:
            idx = np.random.permutation(config.timesteps_per_rollout)
            for start in range(0, config.timesteps_per_rollout, config.minibatches_per_rollout):
                minibatch_idx = idx[start:start + config.minibatches_per_rollout]
                loss = model.compute_loss(ratios, advantages)
                loss.backward()
                optimizer.step()
                #gradient_step()

    # test final model

if __name__ == "__main__":
    main()