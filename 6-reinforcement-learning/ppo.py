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

    def get_action(self, state: Float[torch.Tensor, "batch obs_dim1 obs_dim2 ..."]) -> tuple[Float[torch.Tensor, "batch"], Float[torch.Tensor, "batch"], Float[torch.Tensor, "batch"]]:
        state = self.forward(state)
        logits = self.policy(state)
        probs = torch.softmax(logits, dim=1)

        action = torch.multinomial(probs, num_samples=1).squeeze(-1)
        logp = torch.log(probs.gather(dim=1, index=action.unsqueeze(-1))).squeeze(-1)
        value = self.value(state).squeeze(-1)

        return action, logp, value
    
    def evaluate_action(self, state: Float[torch.Tensor, "batch obs_dim1 obs_dim2 ..."], action: Float[torch.Tensor, "batch"]) -> tuple[Float[torch.Tensor, "batch"], Float[torch.Tensor, "batch"]]:
        state = self.forward(state)
        logits = self.policy(state)
        probs = torch.softmax(logits, dim=1)

        logp = torch.log(probs.gather(dim=1, index=action.unsqueeze(-1))).squeeze(-1)
        value = self.value(state).squeeze(-1)

        return logp, value

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
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), config.lr)

    total_timesteps = 0
    while total_timesteps < config.total_timesteps:
        print("next rollout:")

        # collect TIMESTEPS_PER_ROLLOUT
        states = torch.empty(config.timesteps_per_rollout, obs_dim, device=device)
        actions = torch.empty(config.timesteps_per_rollout, dtype=torch.long, device=device)
        rewards = torch.empty(config.timesteps_per_rollout, device=device)
        logp_old = torch.empty(config.timesteps_per_rollout, device=device)
        values_old = torch.empty(config.timesteps_per_rollout + 1, device=device)
        mask = torch.empty(config.timesteps_per_rollout, device=device)
        with torch.no_grad():
            for t in range(config.timesteps_per_rollout):
                action, logp, value = model.get_action(obs)
                next_obs, reward, terminated, truncated, info = env.step(action.item())

                states[t] = torch.as_tensor(obs, device=device)
                actions[t] = action.item()
                logp_old[t] = logp
                values_old[t] = value
                rewards[t] = reward

                if terminated or truncated:
                    mask[t] = 0
                    obs, info = env.reset()
                else:
                    mask[t] = 1
                    obs = next_obs
                
                total_timesteps += 1
            _, _, value = model.get_action(obs)
            values_old[config.timesteps_per_rollout] = value

            advantages = torch.empty(config.timesteps_per_rollout, device=device)
            advantage_next = 0
            for t in range(config.timesteps_per_rollout-1, -1, -1):
                delta = rewards[t] + config.gamma * mask[t] * values_old[t+1] - values_old[t]
                advantages[t] = delta + config.gamma * config.lam * mask[t] * advantage_next
                advantage_next = advantages[t]

        # update policy
        for epoch in range(config.epochs_per_rollout):
            idx = torch.randperm(config.timesteps_per_rollout, device=device)
            batch_size = config.timesteps_per_rollout // config.minibatches_per_rollout
            for start in range(0, config.timesteps_per_rollout, batch_size):
                optimizer.zero_grad()
                batch_idx = idx[start:start + batch_size]

                batch_actions = actions[batch_idx]
                batch_states = states[batch_idx]
                batch_logp_old = logp_old[batch_idx]
                batch_values_old = values_old[batch_idx]
                batch_advantages = advantages[batch_idx]

                batch_logp, batch_value = model.evaluate_action(batch_states, batch_actions)

                clip_loss = model.compute_clip_loss(
                    logp=batch_logp,
                    logp_old=batch_logp_old,
                    advantage=batch_advantages
                )
                vf_loss = model.compute_vf_loss(
                    value=batch_value,
                    advantage=batch_advantages,
                    value_old=batch_values_old
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