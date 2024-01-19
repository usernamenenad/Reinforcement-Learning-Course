import os.path
from abc import ABC, abstractmethod
from warnings import filterwarnings

from alive_progress import alive_bar

from blackjack.game import Game
from blackjack.info import Info
from blackjack.policy import EpsGreedyPolicy
from blackjack.utils import Q


class MonteCarlo(ABC):
    """
    An interface for all Monte Carlo algorithms.
    """

    @abstractmethod
    def __init__(self, q: Q, gamma: float, alpha: float):
        self.q = q
        self.gamma = gamma
        self.alpha = alpha

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
        filterwarnings("ignore", category=DeprecationWarning)
        print("Starting Incremental Monte Carlo...")

        if os.path.exists("game_log_imc.txt"):
            os.remove("game_log_imc.txt")

        with alive_bar(total=iterations) as bar:
            for i in range(iterations):
                # Play a game
                game.play(EpsGreedyPolicy(epsilon=0.1), self.q, self.gamma)

                # Log game information in a text file
                Info.log_game(game, i, "imc")

                for player in game.players:
                    for rnd in player.experiences:
                        for part in player.experiences[rnd]:
                            s, a, g = part[0], part[1], part[2]
                            self.q[s, a] = (1 - self.alpha) * self.q[s, a] + self.alpha * g

                        # This DOES NOT mean that we're going to forget experiences.
                        # We only clear experiences for the next game.
                        player.experiences[rnd].clear()

                bar()

        print("Finished Incremental Monte Carlo!")
        return self.q
