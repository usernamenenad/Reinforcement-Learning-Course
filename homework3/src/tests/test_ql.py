from blackjack import *


def test_q_learning():
    Player.no_players = 0
    no_players = 2
    players = [Player() for _ in range(no_players)]
    game = Game(players)

    ql = QLearning(q=Q(), gamma=0.9)
    game.attach(ql)
    q = ql.run(game, 20000)

    Info.log_optimal_policy(q, "ql")
