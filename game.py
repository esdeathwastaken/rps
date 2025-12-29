import random
import torch
from utils import json_read
import torch.optim as optim
import matplotlib.pyplot as plt
from agent import PolicyNet, calc_reward, select_action_masked, get_state, IDX_TO_ACTION


class Character:
    def __init__(self, name="Character"):
        self.name = name
        self.max_hp = random.randint(19, 25)
        self.max_armor = random.randint(4, 10)
        self.hp, self.armor = self.max_hp, self.max_armor
        self.attack = {
            "rock": random.randint(10,14),
            "paper": random.randint(3,6),
            "scissors": random.randint(8,11)
        }
        self.defense = {
            "rock": random.randint(6,10),
            "paper": random.randint(2,5),
            "scissors": random.randint(7,11)
        }
        self.charges = {k: 3 for k in ["rock", "paper", "scissors"]}

    def is_alive(self):
        return self.hp > 0


class Enemy(Character):
    def __init__(self, x):
        super().__init__(name=f"Enemy#{x}")
        self.enemy_data = json_read('enemies.json')
        self.generate_enemy_stats(x)

    def generate_enemy_stats(self, x):
        if random.random() < 0.05:
            self.generate_random_enemy()
            return Enemy
        stats = self.extract_enemy_from_file(x)
        setattr(self, 'hp', stats['hp'])
        setattr(self, 'max_hp', stats['hp'])
        setattr(self, 'armor', stats['armor'])
        setattr(self, 'max_armor', stats['armor'])
        setattr(self, "attack", {
            "rock": stats["attack"]["rock"],
            "paper": stats["attack"]["paper"],
            "scissors": stats["attack"]["scissors"]
        })
        setattr(self, "defense", {
            "rock": stats["defense"]["rock"],
            "paper": stats["defense"]["paper"],
            "scissors": stats["defense"]["scissors"]
        })
        return Enemy

    def extract_enemy_from_file(self, x):
        keys = []
        for i in self.enemy_data.keys():
            keys.append(i)
        key = keys[x]
        return self.enemy_data[key]

    def generate_random_enemy(self):
        self.enemy_data = 1
        pass


class Game:
    def __init__(self, x):
        self.player = Character()
        self.enemy = Enemy(x)
        self.stagnation_steps = 0
        self.upgrade_file = json_read('upgrade.json')
        self.rarities = ['common', 'uncommon', 'rare', 'epic', 'legendary']

    def refill_charges(self, player_action, enemy_action):
        for action in self.player.charges:
            if self.player.charges[action] < 3 and action != player_action:
                self.player.charges[action] += 1
        for action in self.enemy.charges:
            if self.enemy.charges[action] < 3 and action != enemy_action:
                self.enemy.charges[action] += 1

    def game_step(self, model, optimizer):
        old_data = (self.player.hp, self.player.armor, self.enemy.hp, self.enemy.armor, self.player.charges.copy())
        state = get_state(self)
        action_idx, log_prob = select_action_masked(model, state, self.player)
        p_act = IDX_TO_ACTION[action_idx]
        e_act = random.choice([a for a, c in self.enemy.charges.items() if c > 0])
        self.player.charges[p_act] -= 1
        self.enemy.charges[e_act] -= 1
        logic = GameLogic()
        result = logic.rps_result(p_act, e_act)
        logic.handle_result(result, self.player, self.enemy, p_act, e_act)

        reward = calc_reward(*old_data, self)


        loss = -log_prob * (reward / 10.0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        self.refill_charges(p_act, e_act)
        return reward

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
            loot_choice['rarity'] = loot
            loot_list.append(loot_choice)
        added_weapons = self.choose_random_weapon(loot_list)
        return added_weapons

    @staticmethod
    def get_key_from_dict(dict_):
        for i, v in dict_.items():
            return i, v

    @staticmethod
    def choose_random_weapon(data):
        for i in data:
            if 'attack' in i or 'defense' in i:
                i['weapon'] = random.choice(['rock', 'paper', 'scissors'])
        return data

    def increase_stats(self, data, value, key):
        weapon = data.get('weapon')
        if weapon is not None:
            stat = getattr(self.player, key)
            stat[weapon] += value
            #print(f'{weapon} {key} увеличено на {value}')
        elif key == 'hp_max':
            self.player.max_hp += value
            self.player.hp += value
            #print(f'максимальное хп увеличено на {value}')
        elif key == 'armor':
            self.player.max_armor += value
            # print(f'армор увеличен на {value}')
        elif key == 'hp_heal':
            self.player.hp += value
            if self.player.hp > self.player.max_hp:
                self.player.hp = self.player.max_hp
        #     print(f'здоровье восстановлено на {value}')
        # print('фаза прокачки завершена')

    def upgrade(self):
        random_rarities = self.random_rarities_choices()
        loot_list = self.exclude_repeats(random_rarities)
        best_choice = self.get_score_from_data(loot_list)
        key, value = self.get_key_from_dict(best_choice)
        self.increase_stats(best_choice, key=key, value=value)

    def get_score_from_data(self, data):
        current_score = 0
        current_hp_percent = (self.player.hp / self.player.max_hp) * 100
        score_table = {
            "common": 1,
            "uncommon": 2,
            "rare": 3,
            "epic": 7,
            "legendary": 12,
            "armor": 2.5,
            "hp_max": 1.5,
            "hp_heal": 0,
            "attack": {
                "rock": 2,
                "paper": 1,
                "scissors": 2
            },
            "defense": {
                "rock": 2,
                "paper": 1,
                "scissors": 2
            },
        }
        score_list = []
        for i in data:
            for x in i:
                if x in ('attack', 'defense'):
                    current_score += score_table[x][i['weapon']]
                elif x in ('armor', 'hp_max'):
                    current_score += score_table[x]
                elif x == 'hp_heal':
                    if 0 <= current_hp_percent <= 74:
                        current_score += 99
                    elif 75 <= current_hp_percent <= 90 and self.player.armor <= 2:
                        current_score += 2
            current_score += score_table[i['rarity']]
            score_list.append(current_score)
            current_score = 0
        best_choice = max(score_list)
        best_choice_index = score_list.index(best_choice)
        return data[best_choice_index]




class GameLogic:
    WIN_MAP = {("rock", "scissors"): 1, ("paper", "rock"): 1, ("scissors", "paper"): 1}

    def rps_result(self, a, b):
        if a == b: return 0
        return 1 if (a, b) in self.WIN_MAP else -1

    @staticmethod
    def apply_damage(attacker, defender, atk_type, def_type):
        damage = attacker.attack[atk_type]
        block = defender.defense[def_type]
        raw_damage = damage - block
        if raw_damage > 0:
            absorbed = min(defender.armor, raw_damage)
            defender.armor -= absorbed
            final_hp_damage = raw_damage - absorbed
            defender.hp -= final_hp_damage
            defender.hp = max(defender.hp, 0)
        # print(
        #     f"[{attacker.__class__.__name__}] ➜ [{defender.__class__.__name__}] | "
        #     f"{atk_type.upper()} vs {def_type.upper()} | "
        #     f"Урон: {damage} (Защита: {block}) | "
        #     f"Броня поглотила: {absorbed}, По HP: -{final_hp_damage} | "
        #     f"Осталось HP: {defender.hp}",
        #     f"Осталось ARM: {defender.armor}"
        # )

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
    model = PolicyNet()
    try:
        model.load_state_dict(torch.load('best_policy.pth'))
        print("Веса загружены. Продолжаем обучение!")
    except:
        print("Новая модель. Начинаем с нуля.")
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    history = []
    best_score = float('-inf')
    for ep in range(1000):
        total_ep_reward = 0
        for x in range(12):
            game = Game(x)
            steps = 0
            stagnation_counter = 0
            last_stats = (game.player.hp, game.player.armor, game.enemy.hp, game.enemy.armor)
            while game.player.is_alive() and game.enemy.is_alive():
                if steps > 150:
                    break
                reward = game.game_step(model, optimizer)
                total_ep_reward += reward
                steps += 1
                current_stats = (game.player.hp, game.player.armor, game.enemy.hp, game.enemy.armor)
                if current_stats == last_stats:
                    stagnation_counter += 1
                else:
                    stagnation_counter = 0
                    last_stats = current_stats
                if stagnation_counter >= 20:
                    game.enemy.hp = 0
                    total_ep_reward += 5
                    break
            if not game.enemy.is_alive() and game.player.is_alive():
                game.upgrade()
        if total_ep_reward > best_score:
            best_score = total_ep_reward
            torch.save(model.state_dict(), 'best_policy.pth')
            print(f"Эпизод {ep}: Новый рекорд! {best_score:.2f}. Модель сохранена.")
        history.append(total_ep_reward)
        if ep % 100 == 0 and ep > 0:
            plt.plot(history)
            plt.savefig('progress.png')
            plt.clf()

    plt.plot(history)
    plt.show()