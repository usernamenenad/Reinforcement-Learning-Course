from dataclasses import field

from .blackjack import *


@dataclass
class Q:
    """
    Class to represent Q estimates.
    """

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


class MonteCarlo(ABC):
    """
    An interface for all Monte Carlo algorithms.
    """

    def __init__(self, q: Q, gamma: float, alpha: float, iterations: int):
        self.q = q
        self.gamma = gamma
        self.alpha = alpha
        self.iterations = iterations

    @abstractmethod
    def run(self, game: Game) -> None:
        pass


class IncrMonteCarlo(MonteCarlo):

    def __init__(self, q: Q = None, gamma: float = 1.0, alpha: float = 0.5, iterations: int = 1000):
        super().__init__(q if q else Q(), gamma, alpha, iterations)

    def run(self, game: Game) -> Q:
        for _ in range(self.iterations):
            game.play(self.gamma)

            # What (state, action) pairs showed up in this game.
            # We will use the `every occurrence` approach, meaning that
            # we will average all gains and not only count the first one.
            occurrences: dict[tuple[PlayerState, Action], list[float]] = dict()

            for player in game.players:
                for round in player.experiences:
                    for part in player.experiences[round]:
                        s, a, g = part[0], part[1], part[2]
                        if (s, a) not in occurrences:
                            occurrences[(s, a)] = list()
                        occurrences[(s, a)].append(g)

                    # This DOES NOT mean that we're going to forget (clear)
                    # the experiences. All experiences of this game are
                    # transferred to `occurrences` dictionary, and we
                    # clear experiences for the next game.
                    player.experiences[round].clear()

            for sa in occurrences:
                average = sum(occurrences[sa]) / len(occurrences[sa])
                self.q[sa] = self.q[sa] + self.alpha * (average - self.q[sa])

        return self.q
