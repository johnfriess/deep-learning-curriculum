import torch
import hydra
import gymnasium as gym
import numpy as np
from gymnasium.spaces.utils import flatdim
from jaxtyping import Int, Float

class PPO(torch.nn.Module):
    def __init__(self, obs_dim=4, hidden_dim=64, action_dim=2):
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

    def get_action(self, state: Float[torch.Tensor, "batch obs_dim1 obs_dim2 ..."]) -> Float[torch.Tensor, "batch action"]:
        logits = self.get_policy(state)
        probs = torch.softmax(logits, dim=1)
        return torch.multinomial(probs, num_samples=1)
    
    def get_value(self, state: Float[torch.Tensor, "batch obs_dim1 obs_dim2 ..."]) -> Float[torch.Tensor, "batch value"]:
        state = self.forward(state)
        return self.value(state)

    def compute_loss(self):
        pass

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(config):
    env = gym.make(config.env, render_mode="human")
    obs_dim = flatdim(env.observation_space)
    action_dim = flatdim(env.action_space)

    model = PPO(
        obs_dim=obs_dim,
        hidden_dim=64,
        action_dim=action_dim 
    )
    optimizer = torch.optim.Adam(model.parameters(), config.lr)

    obs = []
    advantages = []


    total_timesteps = 0
    while total_timesteps < config.total_timesteps:
        # collect TIMESTEPS_PER_ROLLOUT
        # collect X number of trajectories under current polic
        for rollout_step in range(config.timesteps_per_rollout):
            

            total_timesteps += 1
        
        # update policy
        for epoch in config.epochs_per_rollout:
            idx = np.random.permutation(config.timesteps_per_rollout)
            for start in range(0, config.timesteps_per_rollout, config.minibatches_per_rollout):
                minibatch_idx = idx[start:start + config.minibatches_per_rollout]
                #gradient_step()

    # test final model

if __name__ == "__main__":
    main()