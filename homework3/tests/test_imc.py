from blackjack import *


def test_incr_monte_carlo():
    Player.no_players = 0
    no_players = 2
    players = [Player() for _ in range(no_players)]
    dealer = Dealer()
    game = Game(players, dealer)

    imc = IncrMonteCarlo(q=Q(), gamma=0.9)
    q = imc.run(game, iterations=20000)

    Info.log_optimal_policy(q, "imc")
    Info.log_q_values(q, "imc")
