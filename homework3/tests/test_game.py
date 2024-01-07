from blackjack import *


def test_game():
    no_players = 3
    players = [Player() for _ in range(no_players)]
    game = Game(players)

    game.play(policy=RandomPolicy(), gamma=0.9, q=Q())
    Info.draw_experience(game, rnd=2)

    Info.log_experiences(game.players)
