import os.path

from tqdm import tqdm, trange
import warnings

from .info import *


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
    """
    A incremental Monte Carlo algorithm.
    """

    def __init__(self, q: Q = None, gamma: float = 1.0, alpha: float = 0.05, iterations: int = 1000):
        super().__init__(q if q else Q(), gamma, alpha, iterations)

    def run(self, game: Game) -> Q:

        warnings.filterwarnings("ignore", category=DeprecationWarning)
        print("Starting Incremental Monte Carlo...")

        if os.path.exists("game_log.txt"):
            os.remove("game_log.txt")

        for i in trange(self.iterations):

            # Log wins, draws and losses for each round of the game
            game.play(self.q, self.gamma)

            # Log game information in a text file
            Info.log_game(game, i)

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

                    # This DOES NOT mean that we're going to forget experiences.
                    # All experiences of this game are
                    # transferred to `occurrences` dictionary, and we
                    # clear experiences for the next game.
                    player.experiences[round].clear()

            for sa in occurrences:
                average = sum(occurrences[sa]) / len(occurrences[sa])
                self.q[sa] = self.q[sa] + self.alpha * (average - self.q[sa])

        print("Finished Incremental Monte Carlo!")
        return self.q
