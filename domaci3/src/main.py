from blackjack.info import *

has_ace = [True, False]

no_players = 10

# dealer = Dealer()
players = [Player() for _ in range(no_players)]

game = Game(players)

game.play(gamma=0.9)
Info.draw_experience(game, 1)
plt.show()
