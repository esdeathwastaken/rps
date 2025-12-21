import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

IDX_TO_ACTION = {
    0: "rock",
    1: "paper",
    2: "scissors"}

class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(27, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.net(x)


def get_state(game):
    p, e = game.player, game.enemy
    state = [
        p.hp, p.max_hp, p.armor, p.max_armor,
        e.hp, e.max_hp, e.armor, e.max_armor,
    ]
    for a in ["rock", "paper", "scissors"]:
        state += [p.attack[a], p.defense[a], p.charges[a]]
    for a in ["rock", "paper", "scissors"]:
        state += [e.attack[a], e.defense[a], e.charges[a]]
    return np.array(state, dtype=np.float32)


def select_action_masked(model, state, player):
    state = torch.tensor(state)
    probs = model(state)

    mask = torch.tensor([
        player.charges["rock"] > 0,
        player.charges["paper"] > 0,
        player.charges["scissors"] > 0
    ], dtype=torch.float32)

    probs = probs * mask
    if probs.sum() == 0:
        probs = torch.ones(3) / 3
    else:
        probs = probs / probs.sum()

    dist = torch.distributions.Categorical(probs)
    action = dist.sample()
    return action.item(), dist.log_prob(action)


def calc_reward(old_p_hp, old_e_hp, game):
    reward = 0
    reward += (old_p_hp - game.player.hp) * -0.2
    reward += (old_e_hp - game.enemy.hp) * 0.5
    if not game.enemy.is_alive():
        reward += 30
    if not game.player.is_alive():
        reward -= 50
    return reward
