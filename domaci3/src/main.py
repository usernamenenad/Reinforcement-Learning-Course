from copy import deepcopy

from blackjack.montecarlo import *

has_ace = [True, False]

no_players = 2

players = [Player() for _ in range(no_players)]

game = Game(players)

q = Q()
qq = deepcopy(q.q)
# print(q.q)
algorithm = IncrMonteCarlo()
algorithm.run(game=game, q=q, gamma=0.9, iter=100)

for key in q.q:
    if q.q[key] != qq[key]:
        print(f"{key}, {q.q[key]}")
