import os.path
from abc import ABC, abstractmethod
from typing import Optional
from warnings import filterwarnings

from alive_progress import alive_bar

from blackjack.agents import Player
from blackjack.game import Game
from blackjack.info import Info
from blackjack.utils import State, Action, Q


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
    def run(self, game: Game, iterations: int = 1000) -> Q:
        pass


class IncrMonteCarlo(MonteCarlo):
    """
    A incremental Monte Carlo algorithm.
    """

    def __init__(
        self, q: Q | None = None, gamma: float = 1.0, alpha: float = 0.05
    ) -> None:
        super().__init__(q if q is not None else Q(), gamma, alpha)

    def run(self, game: Game, iterations: int = 1000) -> Q:
        filterwarnings("ignore", category=DeprecationWarning)
        print("Starting Incremental Monte Carlo...")

        if os.path.exists("game_log_imc.txt"):
            os.remove("game_log_imc.txt")

        with alive_bar(total=iterations) as bar:
            for i in range(iterations):
                # Play a game
                game.play(self.q, self.gamma)

                # Log game information in a text file
                Info.log_game(game, i, "imc")

                for player in game.players:
                    for rnd in player.experiences:
                        for part in player.experiences[rnd]:
                            s: State = part[0]
                            a: Action = part[1]
                            g: float = part[2]
                            self.q[s, a] = (1 - self.alpha) * self.q[
                                s, a
                            ] + self.alpha * g

                        # This DOES NOT mean that we're going to forget experiences.
                        # We only clear experiences for the next game.
                        player.experiences[rnd].clear()

                bar()

        print("Finished Incremental Monte Carlo!")
        return self.q
