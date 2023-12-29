from blackjack import *
import multiprocessing as mp

Player.no_players = 0
no_players = 2


def imc():
    players_imc = [Player() for _ in range(no_players)]
    game_imc = Game(players_imc)
    q_imc = Q()

    imc = IncrMonteCarlo(q=q_imc, gamma=0.9)
    imc.run(game_imc, iterations=20000)

    Info.log_optimal_policy(q_imc, "mc")


def ql():
    players_ql = [Player() for _ in range(no_players)]
    game_ql = Game(players_ql)
    q_ql = Q()

    ql = QLearning(q=q_ql, gamma=0.9)
    game_ql.attach(ql)
    ql.run(game_ql, iterations=20000)

    Info.log_optimal_policy(q_ql, "ql")


if __name__ == '__main__':
    p1 = mp.Process(target=imc)
    p2 = mp.Process(target=ql)

    p1.start()
    p2.start()
