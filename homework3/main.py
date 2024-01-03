import multiprocessing as mp

from blackjack import *

Player.no_players = 0
no_players = 2


def imc_func():
    players_imc = [Player() for _ in range(no_players)]
    game_imc = Game(players_imc)
    q_imc = Q()

    imc = IncrMonteCarlo(q=q_imc, gamma=0.9)
    imc.run(game_imc, iterations=20000)

    Info.log_optimal_policy(q_imc, "imc")
    Info.log_q_values(q_imc, "imc")


def ql_func():
    players_ql = [Player() for _ in range(no_players)]
    game_ql = Game(players_ql)
    q_ql = Q()

    ql = QLearning(q=q_ql, gamma=0.9)
    game_ql.attach(ql)
    ql.run(game_ql, iterations=20000)

    Info.log_optimal_policy(q_ql, "ql")
    Info.log_q_values(q_ql, "ql")


def sarsa_func():
    players_sarsa = [Player() for _ in range(no_players)]
    game_sarsa = Game(players_sarsa)
    q_sarsa = Q()

    sarsa = SARSA(q=q_sarsa, gamma=0.9)
    game_sarsa.attach(sarsa)
    sarsa.run(game_sarsa, iterations=20000)

    Info.log_optimal_policy(q_sarsa, "sarsa")
    Info.log_q_values(q_sarsa, "sarsa")


if __name__ == '__main__':
    p1 = mp.Process(target=imc_func)
    p2 = mp.Process(target=ql_func)
    p3 = mp.Process(target=sarsa_func)

    p1.start()
    p2.start()
    p3.start()
