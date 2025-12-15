import random
import json


def json_read():
    with open("upgrade.json", "r") as f:
        upgrade_dict = json.load(f)
    return upgrade_dict


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

    def is_alive(self):
        return self.hp > 0


class Game:
    def __init__(self):
        self.player = Character()
        self.enemy = Character()
        self.upgrade_file = json_read()
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

    def game_step(self):
        player_action = self.random_action(self.player)
        enemy_action = self.random_action(self.enemy)
        logic = GameLogic()
        result = logic.rps_result(player_action, enemy_action)
        print(f"Player chooses {player_action}, Enemy chooses {enemy_action}, Result: {result}")
        logic.handle_result(result, self.player, self.enemy, player_action, enemy_action)
        self.refill_charges(player_action, enemy_action)
        print(f"Player HP: {self.player.hp}, Armor: {self.player.armor}")
        print(f"Enemy HP: {self.enemy.hp}, Armor: {self.enemy.armor}")

    def random_rarities_choices(self):
        loot_list = []
        weights = [40, 30, 15, 10, 5]
        for loot in range(0, 3):
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
        for i, _ in dict_.items():
            return i, _

    @staticmethod
    def choose_random_weapon(key):
        if key == 'attack' or key == 'defense':
            weapon = random.choice(['rock', 'scissors', 'paper'])
            return weapon

    def increase_stats(self, key, value, weapon):
        if weapon is not None:
            stat = getattr(self.player, key)
            stat[weapon] += value
            print(f'{weapon} {key} увеличено на {value}')
        elif key == 'hp_max':
            self.player.max_hp += value
            print(f'максимальное хп увеличено на {value}')
        elif key == 'armor':
            self.player.max_armor += value
            print(f'армор увеличен на {value}')
        elif key == 'hp_heal':
            self.player.hp += value
            if self.player.max_hp < self.player.hp:
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
    def apply_damage(who, to_whom, atk_type, def_type):
        damage = who.attack[atk_type]
        block = to_whom.defense[def_type]
        effective_damage = max(0, damage - block)
        if to_whom.armor > 0:
            if to_whom.armor >= effective_damage:
                to_whom.armor -= effective_damage
                effective_damage = 0
            else:
                effective_damage -= to_whom.armor
                to_whom.armor = 0
        to_whom.hp -= effective_damage

    @staticmethod
    def armor_regen(action, who):
        amount_regen = who.defense[action]
        who.armor += amount_regen
        if who.max_armor <= who.armor:
            who.armor = who.max_armor

    def handle_result(self, result, player, enemy, player_action, enemy_action):
        if result == 1:
            self.apply_damage(player, enemy, player_action, enemy_action)
            self.armor_regen(player_action, player)
        elif result == -1:
            self.apply_damage(enemy, player, enemy_action, player_action)
            self.armor_regen(enemy_action, enemy)
        elif result == 0:
            self.apply_damage(player, enemy, enemy_action, player_action)
            self.apply_damage(enemy, player, enemy_action, player_action)
            self.armor_regen(player_action, player)
            self.armor_regen(enemy_action, enemy)


if __name__ == '__main__':
    for _ in range(1, 10):
        game = Game()
        while game.player.is_alive() and game.enemy.is_alive():
            game.game_step()
            if not game.enemy.is_alive():
                game.upgrade()
