import torch
import torch.nn as nn
import numpy as np

IDX_TO_ACTION = {0: "rock", 1: "paper", 2: "scissors"}


class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(26, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.net(x)

    def save(self, path="battle_model.pth"):
        torch.save(self.state_dict(), path)
        print(f"Модель сохранена в {path}")

    def load(self, path="battle_model.pth"):
        try:
            self.load_state_dict(torch.load(path))
            print(f"Модель загружена из {path}")
        except FileNotFoundError:
            print("Файл модели не найден, начинаем с нуля.")


def get_state(game):
    p, e = game.player, game.enemy
    state = [p.hp, p.max_hp, p.armor, p.max_armor, e.hp, e.max_hp, e.armor, e.max_armor]
    for a in ["rock", "paper", "scissors"]:
        state += [p.attack[a], p.defense[a], p.charges[a]]
    for a in ["rock", "paper", "scissors"]:
        state += [e.attack[a], e.defense[a], e.charges[a]]
    return np.array(state, dtype=np.float32)


def select_action_masked(model, state, player):
    state = torch.tensor(state, dtype=torch.float32)
    probs = model(state)
    mask = torch.tensor([player.charges[a] > 0 for a in ["rock", "paper", "scissors"]], dtype=torch.float32)
    masked_probs = probs * mask
    if masked_probs.sum() <= 0:
        masked_probs = mask / (mask.sum() + 1e-8)
    else:
        masked_probs = masked_probs / masked_probs.sum()

    dist = torch.distributions.Categorical(masked_probs)
    action = dist.sample()
    return action.item(), dist.log_prob(action)


def calc_reward(old_p_hp, old_p_arm, old_e_hp, old_e_arm, old_charges, game):
    reward = 0.0
    reward += (old_e_hp - game.enemy.hp) * 1.5
    reward += (old_e_arm - game.enemy.armor) * 0.5
    reward -= (old_p_hp - game.player.hp) * 2.0
    reward -= (old_p_arm - game.player.armor) * 0.5
    for i in ['rock', 'paper', 'scissors']:
        if game.player.charges[i] <= 0:
            reward -= 10
    if not game.enemy.is_alive(): reward += 20
    if not game.player.is_alive(): reward -= 30
    return reward