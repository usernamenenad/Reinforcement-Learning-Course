from blackjack import *
import multiprocessing as mp

if __name__ == '__main__':
    players_imc = [Player() for _ in range(2)]
    game_imc = Game(players=players_imc)
    q_imc = Q()

    players_ql = [Player() for _ in range(2)]
    game_ql = Game(players=players_ql)
    q_ql = Q()

    imc = IncrMonteCarlo(q=q_imc, gamma=0.9)
    ql = QLearning(q=q_ql, gamma=0.9)

    p1 = mp.Process(target=imc.run, args=(game_imc, 20000,))
    p2 = mp.Process(target=ql.run, args=(game_ql, 20000,))

    p1.start()
    p2.start()

    Info.log_optimal_policy(q_imc, "mc")
    Info.log_optimal_policy(q_ql, "ql")
