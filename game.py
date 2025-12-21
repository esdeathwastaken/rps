import random
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from agent import PolicyNet
from agent import calc_reward
from agent import select_action_masked
from agent import get_state
from agent import IDX_TO_ACTION

def json_read():
    with open("upgrade.json", "r") as f:
        return json.load(f)


class Character:
    def __init__(self):
        self.max_hp = random.randint(16, 25)
        self.max_armor = random.randint(0, 10)
        self.hp = self.max_hp
        self.armor = self.max_armor
        self.attack = {
            "rock": random.randint(3, 12),
            "paper": random.randint(3, 12),
            "scissors": random.randint(3, 12),
        }

        self.defense = {
            "rock": random.randint(0, 8),
            "paper": random.randint(0, 8),
            "scissors": random.randint(0, 8),
        }

        self.charges = {
            "rock": 3,
            "paper": 3,
            "scissors": 3,
        }

    def __str__(self):
        return self.__class__.__name__

    def is_alive(self):
        return self.hp > 0

class Enemy(Character):
    def __init__(self):
        super().__init__()
        self.max_hp = 5
        self.max_armor = 3
        self.hp = self.max_hp
        self.armor = self.max_armor
        self.attack = {
            "rock": 2,
            "paper": 2,
            "scissors": 2,
        }

        self.defense = {
            "rock": 2,
            "paper": 2,
            "scissors": 2,
        }

        self.charges = {
            "rock": 3,
            "paper": 3,
            "scissors": 3,
        }

    def statistic_boost(self):
        self.max_hp += random.randint(1,2)
        self.max_armor += random.randint(0,1)
        self.hp = self.max_hp
        self.armor = self.max_armor
        for key in self.attack:
            self.attack[key] += random.randint(0, 2)
        for key in self.defense:
            self.defense[key] += random.randint(0, 1)

    def __str__(self):
        return self.__class__.__name__

class Game:
    def __init__(self):
        self.player = Character()
        self.enemy = Enemy()
        self.upgrade_file = json_read()
        self.stage = 'battle'
        self.rarities = ['common', 'uncommon', 'rare', 'epic', 'legendary']

    @staticmethod
    def random_action(player):
        available = [action for action, charges in player.charges.items() if charges > 0]
        if not available:
            return None
        action = random.choice(available)
        if player.charges[action] > 1:
            player.charges[action] -= 1
        else:
            player.charges[action] = -1
        return action

    def refill_charges(self, player_action, enemy_action):
        for action in self.player.charges:
            if self.player.charges[action] < 3 and action != player_action:
                self.player.charges[action] += 1
        for action in self.enemy.charges:
            if self.enemy.charges[action] < 3 and action != enemy_action:
                self.enemy.charges[action] += 1

    def game_step(self, model, optimizer):
        old_p_hp = self.player.hp
        old_e_hp = self.enemy.hp
        state = get_state(self)
        action_idx, log_prob = select_action_masked(model, state, self.player)
        player_action = IDX_TO_ACTION[action_idx]
        enemy_action = self.random_action(self.enemy)
        logic = GameLogic()
        result = logic.rps_result(player_action, enemy_action)
        print(f"Player chooses {player_action}, Enemy chooses {enemy_action}, Result: {result}")
        logic.handle_result(result, self.player, self.enemy, player_action, enemy_action)
        self.refill_charges(player_action, enemy_action)
        reward = calc_reward(old_p_hp, old_e_hp, self)
        loss = -log_prob * reward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    def random_rarities_choices(self):
        loot_list = []
        weights = [40, 30, 15, 10, 5]
        for _ in range(3):
            loot = random.choices(self.rarities, weights=weights, k=1)[0]
            loot_list.append(loot)
        return loot_list

    def exclude_repeats(self, random_rarities):
        loot_list = []
        used_keys = set()
        for loot in random_rarities:
            while True:
                loot_choice = random.choice(list(self.upgrade_file[loot]))
                key = next(iter(loot_choice))
                if key not in used_keys:
                    break
            used_keys.add(key)
            loot_list.append(loot_choice)
        return loot_list

    @staticmethod
    def get_key_from_dict(dict_):
        for i, v in dict_.items():
            return i, v

    @staticmethod
    def choose_random_weapon(key):
        if key in ('attack', 'defense'):
            return random.choice(['rock', 'paper', 'scissors'])

    def increase_stats(self, key, value, weapon):
        if weapon is not None:
            stat = getattr(self.player, key)
            stat[weapon] += value
            print(f'{weapon} {key} увеличено на {value}')
        elif key == 'hp_max':
            self.player.max_hp += value
            self.player.hp += value
            print(f'максимальное хп увеличено на {value}')
        elif key == 'armor':
            self.player.max_armor += value
            print(f'армор увеличен на {value}')
        elif key == 'hp_heal':
            self.player.hp += value
            if self.player.hp > self.player.max_hp:
                self.player.hp = self.player.max_hp
            print(f'здоровье восстановлено на {value}')
        print('фаза прокачки завершена')

    def upgrade(self):
        random_rarities = self.random_rarities_choices()
        loot_list = self.exclude_repeats(random_rarities)
        random_loot = random.choice(loot_list)
        key, value = self.get_key_from_dict(random_loot)
        weapon = self.choose_random_weapon(key)
        self.increase_stats(key, value, weapon)
        self.enemy.statistic_boost()


class GameLogic:
    def __init__(self):
        self.WIN_MAP = {
            ("rock", "scissors"): 1,
            ("paper", "rock"): 1,
            ("scissors", "paper"): 1,
        }

    def rps_result(self, a, b):
        if a == b:
            return 0
        return 1 if (a, b) in self.WIN_MAP else -1

    @staticmethod
    def apply_damage(attacker, defender, atk_type, def_type):
        damage = attacker.attack[atk_type]
        block = defender.defense[def_type]
        effective_damage = max(0, damage - block)
        absorbed = min(defender.armor, effective_damage)
        defender.armor -= absorbed
        effective_damage -= absorbed
        defender.hp -= effective_damage
        defender.hp = max(defender.hp, 0)
        defender.armor = max(defender.armor, 0)
        print(f"  -> {attacker} наносит {damage} урона ({block} блок) | "
              f"броня поглотила: {absorbed}, HP потеряно: {effective_damage}")

    @staticmethod
    def armor_regen(action, who):
        regen = who.defense[action]
        who.armor = min(who.max_armor, who.armor + regen)

    def handle_result(self, result, player, enemy, player_action, enemy_action):
        if result == 1:
            self.apply_damage(player, enemy, player_action, enemy_action)
            self.armor_regen(player_action, player)
        elif result == -1:
            self.apply_damage(enemy, player, enemy_action, player_action)
            self.armor_regen(enemy_action, enemy)
        else:
            self.apply_damage(player, enemy, player_action, enemy_action)
            self.apply_damage(enemy, player, enemy_action, player_action)
            self.armor_regen(player_action, player)
            self.armor_regen(enemy_action, enemy)


if __name__ == '__main__':
    game = Game()
    model = PolicyNet()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    counter = 0
    while counter != 10:
        while game.player.is_alive() and game.enemy.is_alive():
            game.game_step(model, optimizer)
            if not game.player.is_alive():
                print('player is dead')
                break
            if not game.enemy.is_alive():
                print('enemy is dead')
                game.upgrade()
                counter += 1
                print(f'round: {counter}')
        break