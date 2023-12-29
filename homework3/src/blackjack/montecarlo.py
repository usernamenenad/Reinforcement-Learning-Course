import os.path
import time

from tqdm import trange
from alive_progress import alive_bar
import warnings

from .info import *


class MonteCarlo(ABC):
    """
    An interface for all Monte Carlo algorithms.
    """

    @abstractmethod
    def __init__(self, q: Q, gamma: float, alpha: float):
        self.q = q
        self.gamma = gamma
        self.alpha = alpha
        print("just testing")

    @abstractmethod
    def run(self, game: Game) -> None:
        pass


class IncrMonteCarlo(MonteCarlo):
    """
    A incremental Monte Carlo algorithm.
    """

    def __init__(self, q: Q = None, gamma: float = 1.0, alpha: float = 0.05):
        super().__init__(q if q else Q(), gamma, alpha)

    def run(self, game: Game, iterations: int = 1000) -> Q:

        warnings.filterwarnings("ignore", category=DeprecationWarning)
        print("Starting Incremental Monte Carlo...")

        if os.path.exists("game_log_mc.txt"):
            os.remove("game_log_mc.txt")

        with alive_bar(total=iterations) as bar:
            for i in range(iterations):
                # Play a game
                game.play(GreedyPolicy(), self.q, self.gamma)

                # Log game information in a text file
                Info.log_game(game, i, "mc")

                # What (state, action) pairs showed up in this game.
                # We will use the `every occurrence` approach, meaning that
                # we will average all gains and not only count the first one.
                occurrences: dict[tuple[PlayerState, Action], list[float]] = dict()

                for player in game.players:
                    for rnd in player.experiences:
                        for part in player.experiences[rnd]:
                            s, a, g = part[0], part[1], part[2]
                            if (s, a) not in occurrences:
                                occurrences[(s, a)] = list()
                            occurrences[(s, a)].append(g)

                        # This DOES NOT mean that we're going to forget experiences.
                        # All experiences of this game are
                        # transferred to `occurrences` dictionary, and we
                        # clear experiences for the next game.
                        player.experiences[rnd].clear()

                for sa in occurrences:
                    average = sum(occurrences[sa]) / len(occurrences[sa])
                    self.q[sa] = self.q[sa] + self.alpha * (average - self.q[sa])

                bar()

        print("Finished Incremental Monte Carlo!")
        return self.q
