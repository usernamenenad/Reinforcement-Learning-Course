from blackjack import *

has_ace = [True, False]

dealer = Dealer()
players = [Player() for _ in range(2)]

game = Game(dealer, players)

game.play(gamma=0.9)
