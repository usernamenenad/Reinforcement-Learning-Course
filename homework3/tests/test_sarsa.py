from blackjack import *


def test_sarsa():
    Player.no_players = 0
    no_players = 2
    players = [Player() for _ in range(no_players)]
    dealer = Dealer()
    game = Game(players, dealer)

    sarsa = SARSA(q=Q(), gamma=0.9)
    game.attach(sarsa)
    q = sarsa.run(game, 20000)

    Info.log_optimal_policy(q, "sarsa")
    Info.log_q_values(q, "sarsa")
