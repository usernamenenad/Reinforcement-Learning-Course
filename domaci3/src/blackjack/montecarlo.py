from .blackjack import *


@dataclass
class Q:

    def __init__(self):
        all_states = [PlayerState(total=total, has_ace=has_ace, dealer_total=0)
                      for total in range(0, 22)
                      for has_ace in [False, True]]

        all_actions = [Action.HOLD, Action.HIT]

        self.q = {
            (s, a): random()
            for s in all_states
            for a in all_actions
        }

    def __getitem__(self, key: tuple[PlayerState, Action]) -> float:
        return self.q[key]

    def __setitem__(self, key: tuple[PlayerState, Action], gain: float):
        self.q[key] = gain


class IncrMonteCarlo:

    def run(self, game: Game, q: Q, gamma: float = 1.0, alpha: float = 0.05, iter: int = 1000):
        for i in range(iter):
            # print(f"Going into iteration {i}")
            game.play(gamma)
            occured: dict[tuple[PlayerState, Action], list[float]] = dict()
            for player in game.players:
                for round in player.experiences:
                    for part in player.experiences[round]:
                        s, a, g = part[0], part[1], part[2]
                        if (s, a) not in occured:
                            occured[(s, a)] = list()
                        occured[(s, a)].append(g)

                    player.experiences[round].clear()

            for sa in occured:
                average = sum(occured[sa]) / len(occured[sa])
                q[sa] = q[sa] + alpha * (average - q[sa])
