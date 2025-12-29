import torch
import hydra
import gymnasium as gym
from gymnasium.spaces.utils import flatdim
from jaxtyping import Int, Float

class PPO(torch.nn.Module):
    def __init__(self, device=None, obs_dim=4, hidden_dim=64, action_dim=2, epsilon=0.2, entropy_bonus=0.01, c1=0.5, c2=0.01):
        super().__init__()
        self.device = device
        self.epsilon = epsilon
        self.c1 = c1
        self.c2 = c2
        self.entropy_bonus = entropy_bonus
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU()
        )
        self.policy = torch.nn.Linear(hidden_dim, action_dim)
        self.value = torch.nn.Linear(hidden_dim, 1)

    def forward(self, state: Float[torch.Tensor, "batch obs_dim1 obs_dim2 ..."]) -> Float[torch.Tensor, "batch hidden_dim"]:
        if not isinstance(state, torch.Tensor):
            state = torch.as_tensor(state, device=self.device)
        if state.ndim == 1:
            state = state.unsqueeze(0)
        B = state.shape[0]
        state = state.view(B, -1)
        return self.encoder(state)

    def get_policy(self, state: Float[torch.Tensor, "batch obs_dim1 obs_dim2 ..."]) -> Float[torch.Tensor, "batch action_dim"]:
        state = self.forward(state)
        return self.policy(state)

    def get_action(self, state: Float[torch.Tensor, "batch obs_dim1 obs_dim2 ..."]) -> Float[torch.Tensor, "batch action"]:
        state = self.forward(state)
        logits = self.policy(state)
        probs = torch.softmax(logits, dim=1)
        action = torch.multinomial(probs, num_samples=1)
        return action, torch.log(probs.gather(dim=1, index=action)), self.value(state)
    
    def compute_clip_loss(self, logp: Float[torch.Tensor, "t"], logp_old: Float[torch.Tensor, "t"], advantage: Float[torch.Tensor, "t"]) -> Float[torch.Tensor, "t"]:
        ratio = torch.exp(logp - logp_old)
        return torch.minimum(ratio * advantage, torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantage)
    
    def compute_vf_loss(self, value: Float[torch.Tensor, "t"], advantage: Float[torch.Tensor, "t"], value_old: Float[torch.Tensor, "t"]) -> Float[torch.Tensor, "t"]:
        return (value - (advantage + value_old))**2

    def compute_loss(self, clip_loss: Float[torch.Tensor, "t"], vf_loss: Float[torch.Tensor, "t"]) -> Float[torch.Tensor, "loss"]:
        return -torch.mean(clip_loss - self.c1 * vf_loss)

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = gym.make(config.env, render_mode="human")
    obs_dim = flatdim(env.observation_space)
    action_dim = flatdim(env.action_space)

    obs, info = env.reset(seed=0)

    model = PPO(
        device=device,
        obs_dim=obs_dim,
        hidden_dim=64,
        action_dim=action_dim,
        epsilon=config.epsilon
    )
    optimizer = torch.optim.Adam(model.parameters(), config.lr)

    total_timesteps = 0
    while total_timesteps < config.total_timesteps:
        optimizer.zero_grad()

        # collect TIMESTEPS_PER_ROLLOUT
        state = [obs]
        reward = []
        logp_old = []
        value_old = []
        mask = []
        for _ in range(config.timesteps_per_rollout):
            action, logp, v = model.get_action(obs)
            next_obs, r, terminated, truncated, info = env.step(action.item())

            logp_old.append(logp)
            value_old.append(v)
            reward.append(r)
            mask.append(0 if terminated or truncated else 1)
            
            if terminated or truncated:
                obs, info = env.reset()
            else:
                obs = next_obs
            state.append(obs)
            
            total_timesteps += 1

        advantage = [0]
        for t in range(config.timesteps_per_rollout-2, -1, -1):
            delta = reward[t] + config.gamma * mask[t] * value_old[t+1] - value_old[t]
            advantage.append(delta + config.gamma * config.lam * mask[t] * advantage[-1])
        advantage = advantage[::-1]
        advantage.pop()

        state = torch.as_tensor(state, device=device)
        advantage = torch.as_tensor(advantage, device=device)
        logp_old = torch.as_tensor(logp_old, device=device)
        value_old = torch.as_tensor(value_old, device=device)

        # update policy
        for epoch in range(config.epochs_per_rollout):
            idx = torch.randperm(config.timesteps_per_rollout-1)
            for start in range(0, config.timesteps_per_rollout, config.minibatches_per_rollout):
                batch_idx = idx[start:start + config.minibatches_per_rollout]

                batch_state = state[batch_idx]
                batch_logp_old = logp_old[batch_idx]
                batch_value_old = value_old[batch_idx]
                batch_advantage = advantage[batch_idx]

                batch_action, batch_logp, batch_value = model.get_action(batch_state)

                clip_loss = model.compute_clip_loss(
                    logp=batch_logp,
                    logp_old=batch_logp_old,
                    advantage=batch_advantage
                )
                vf_loss = model.compute_vf_loss(
                    value=batch_value,
                    advantage=batch_advantage,
                    value_old=batch_value_old
                )
                loss = model.compute_loss(clip_loss, vf_loss)
                print(f"timestep: {total_timesteps}, epoch: {epoch}, loss: {loss}")
                loss.backward()
                optimizer.step()

    # test final model
    print("testing final model")
    obs, info = env.reset(seed=1)
    done = False
    i = 0
    while not done:
        action, logp, v = model.get_action(obs)
        next_obs, r, terminated, truncated, info = env.step(action.item())
        done = terminated or truncated
        i += 1
        print(f"score: {i}")

if __name__ == "__main__":
    main()