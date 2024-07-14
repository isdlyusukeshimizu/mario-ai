import numpy
import sys
import os
import pickle
from agents.myagent import *
from experiments.episodicexperiment import EpisodicExperiment
from ga.controller import Controller
from tasks.mariotask import MarioTask

class IndividualReward:
    def __init__(self, individual, reward):
        self.individual = individual
        self.reward = reward

    def __str__(self):
        return str(self.reward)

def make_next_generation(experiment, individuals):
    n_individuals = len(individuals)

    rewards = []
    for individual in individuals:
        experiment.agent.individual = individual
        experiment.doEpisodes(1)
        rewards.append(IndividualReward(individual, experiment.task.reward))
        print("reward: {0}".format(experiment.task.reward))

    numberOfElites = 1
    sorted_rewards = sorted(rewards, key=lambda individual_reward: individual_reward.reward, reverse=True)
    print("best reward: {0}".format(sorted_rewards[0].reward))
    elite_individuals = list(map(lambda e: e.individual, sorted_rewards[:numberOfElites]))
    next_individuals = elite_individuals
    while len(next_individuals) < n_individuals:
        father, mother = Controller.select(list(map(lambda individual_reward: individual_reward.individual, rewards)),
                                           list(map(lambda individual_reward: individual_reward.reward, rewards)))
        child1, child2 = Controller.two_points_cross(father, mother)
        next_individuals.append(child1)
        next_individuals.append(child2)
    next_individuals = next_individuals[:n_individuals]
    Controller.mutate(next_individuals, mutation_rate=0.3)
    best_reward = sorted_rewards[0].reward  # 各世代の最高の報酬
    return next_individuals, best_reward

def main():
    agent = MyAgent(None)
    task = MarioTask(agent.name)
    task.env.initMarioMode = 2
    task.env.levelDifficulty = int(sys.argv[1]) if len(sys.argv) == 2 else 0
    experiment = EpisodicExperiment(task, agent)

    n_individuals = 10

    # ランダムな個体を生成。内容の表示はコメントアウトした。
    initial_individuals = [Individual(random=True) for _ in range(n_individuals)]
    # print("Initial individuals:")
    # for i, individual in enumerate(initial_individuals):
    #     print(f"Individual {i}: {individual.data}")

    current_individuals = initial_individuals
    
    n_generations = 100
    all_best_rewards = []  # 各世代の最高の報酬を保存するリストに変更
    for generation in range(n_generations):
        print(f"Generation #{generation} playing...")
        task.env.visualization = generation % 10 == 0
        current_individuals, best_reward = make_next_generation(experiment, current_individuals)
        all_best_rewards.append(best_reward)

        # 各世代の個体の内容を表示。←みにくくなるから、コメントアウトした。
        # print(f"Generation #{generation} individuals:")
        # for i, individual in enumerate(current_individuals):
        #     print(f"Individual {i}: {individual.data}")

        # 各世代ごとに報酬を保存
        save_rewards(all_best_rewards, "best_rewards.pkl")

def save_rewards(rewards, filename):
    file_path = os.path.abspath(filename)
    with open(file_path, "wb") as f:
        pickle.dump(rewards, f)
    print(f"Rewards saved to {file_path}")

if __name__ == "__main__":
    main()
else:
    print("This is module to be run rather than imported.")
